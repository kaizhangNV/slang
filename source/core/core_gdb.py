"""
This python script provides LLDB formatters for Slang core types.
To use it, add the following line to your ~/.lldbinit file:
command script import /path/to/source/core/core_lldb.py
"""


import itertools
import gdb
import sys

if sys.version_info[0] == 3:
	Iterator = object
else:
	class Iterator(object):
		def next(self):
			return type(self).__next__(self)

def default_iterator(val):
	for field in val.type.fields():
		yield field.name, val[field.name]

# Set to True to enable the logger
ENABLE_LOGGING = True

# log to the LLDB formatter stream
def log(msg):
    if ENABLE_LOGGING:
        gdb.formatters.Logger.Logger() >> msg

def make_string(F, L):
    strval = ""
    G = F.uint8
    for X in range(L):
        V = G[X]
        if V == 0:
            break
        strval = strval + chr(V % 256)
    return '"' + strval + '"'

# Return the pointer to the data in a Slang::RefPtr
def get_ref_pointer(valobj):
    return valobj.GetNonSyntheticValue().GetChildMemberWithName("pointer")

# Check if a pointer is nullptr
def is_nullptr(valobj):
    return valobj.GetValueAsUnsigned(0) == 0

# Slang::String summary
def String_summary(valobj, dict):
    buffer_ptr = get_ref_pointer(valobj.GetChildMemberWithName("m_buffer"))
    if is_nullptr(buffer_ptr):
        return '""'
    buffer = buffer_ptr.Dereference()
    length = buffer.GetChildMemberWithName("length").GetValueAsUnsigned(0)
    data = buffer_ptr.GetPointeeData(1, length)
    return make_string(data, length)

# Slang::UnownedStringSlice summary
def UnownedStringSlice_summary(valobj, dict):
    begin = valobj.GetChildMemberWithName("m_begin")
    end = valobj.GetChildMemberWithName("m_end")
    length = end.GetValueAsUnsigned(0) - begin.GetValueAsUnsigned(0)
    if length <= 0:
        return '""'
    data = begin.GetPointeeData(0, length)
    return make_string(data, length)

# Slang::RefPtr synthetic provider
class RefPtr_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return True

    def num_children(self):
        return len(self.children)

    def get_child_index(self, name):
        for index in range(self.num_children()):
            if self.children[index].GetName() == name:
                return index
        return -1

    def get_child_at_index(self, index):
        if index >= 0 and index < self.num_children():
            return self.children[index]
        else:
            return None

    def update(self):
        self.pointer = self.valobj.GetNonSyntheticValue().GetChildMemberWithName(
            "pointer"
        )
        self.children = []
        if not is_nullptr(self.pointer):
            self.children = self.pointer.Dereference().children

# Slang::RefPtr summary
def RefPtr_summary(valobj, dict):
    pointer = valobj.GetNonSyntheticValue().GetChildMemberWithName("pointer")
    if is_nullptr(pointer):
        return "nullptr"
    pointee = pointer.Dereference()
    refcount = pointee.GetChildMemberWithName("referenceCount").GetValueAsUnsigned()
    return str(pointer.GetValue()) + " refcount=" + str(refcount)

# Slang::ComPtr synthetic provider
class ComPtr_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return len(self.children) > 0

    def num_children(self):
        return len(self.children)

    def get_child_index(self, name):
        for index in range(self.num_children()):
            if self.children[index].GetName() == name:
                return index
        return -1

    def get_child_at_index(self, index):
        if index >= 0 and index < self.num_children():
            return self.children[index]
        else:
            return None

    def update(self):
        self.pointer = self.valobj.GetChildMemberWithName("m_ptr")
        self.children = []
        if not is_nullptr(self.pointer):
            self.children = self.pointer.Dereference().children

# Slang::ComPtr summary
def ComPtr_summary(valobj, dict):
    pointer = valobj.GetNonSyntheticValue().GetChildMemberWithName("m_ptr")
    if is_nullptr(pointer):
        return "nullptr"
    return str(pointer.GetValue())

# Slang::Array synthetic provider
class Array_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return True

    def num_children(self):
        return self.count.GetValueAsUnsigned(0)

    def get_child_index(self, name):
        return int(name.lstrip("[").rstrip("]"))

    def get_child_at_index(self, index):
        if index >= 0 and index < self.num_children():
            offset = index * self.data_size
            return self.buffer.CreateChildAtOffset(
                "[" + str(index) + "]", offset, self.data_type
            )
        else:
            return None

    def update(self):
        self.count = self.valobj.GetChildMemberWithName("m_count")
        self.buffer = self.valobj.GetChildMemberWithName("m_buffer")
        self.data_type = self.buffer.GetType().GetArrayElementType()
        self.data_size = self.data_type.GetByteSize()

def ColorVariable(member_name):
    return "\033[36m" + member_name + "\033[0m"

def ColorType(type_name):
    return "\033[33m" + type_name + "\033[0m"

def ColorAddress(address):
    return "\033[34m" + address + "\033[0m"

def IsTypePrintable(type):
    if type.code == gdb.TYPE_CODE_PTR or \
       type.code == gdb.TYPE_CODE_INT or \
       type.code == gdb.TYPE_CODE_FLT or \
       type.code == gdb.TYPE_CODE_CHAR or \
       type.code == gdb.TYPE_CODE_BOOL or \
       type.code == gdb.TYPE_CODE_DECFLOAT or \
       type.code == gdb.TYPE_CODE_ENUM or \
       type.code == gdb.TYPE_CODE_STRING or \
       (type.code == gdb.TYPE_CODE_STRUCT and type.name == "Slang::String"):
        return True


# Slang::String
class StringPrinter:
    def __init__(self, val):
        self.val = val
    def to_string(self):
        buffer = self.val['m_buffer'];
        string = buffer['pointer'];
        if string.address == 0:
            return '""'
        val = (string + 1).cast(gdb.lookup_type('char').pointer());
        return '"' + val.string() + '"';

# Slang::Name
class NamePrinter:
    def __init__(self, val):
        self.val = val
    def to_string(self):
        text = self.val['text']
        textPrinter = StringPrinter(text)
        return textPrinter.to_string()

# Slang::List
class ListPrinter:
    def __init__(self, valobj):
        self.valobj = valobj
        self.Typename = valobj.type.template_argument(0)
        self.count = valobj['m_count']
        self.m_capacity = self.valobj['m_capacity'];
        self.buffer = self.valobj['m_buffer'];

        self.thisTypePrintable = False
        if IsTypePrintable(self.Typename):
            self.thisTypePrintable = True

    def has_children(self):
        return True

    def num_children(self):
        return self.count.GetValueAsUnsigned(0)

    def get_child_index(self, name):
        return int(name.lstrip("[").rstrip("]"))

    def get_child_at_index(self, index):
        if index >= 0 and index < self.num_children():
            offset = index * self.data_size
            return self.buffer.CreateChildAtOffset(
                "[" + str(index) + "]", offset, self.data_type
            )
        else:
            return None

    class _iterator():
        def __init__(self, valobj):
            if valobj is None:
                self.valobj = None
            else:
                self.valobj = valobj
                self.elementCount = self.valobj['m_count']
                self.buffer = self.valobj['m_buffer']
                self.Typename = self.buffer.type
            self.curIndex = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.valobj is None:
                raise StopIteration

            index = self.curIndex
            self.curIndex = self.curIndex + 1
            if self.curIndex > self.elementCount:
                raise StopIteration

            indent = "  "
            prefixStr = '\n' + indent + indent + indent
            firstStr = prefixStr + '[%d]' % index

            val = self.buffer[index]
            secondStr = prefixStr + '[%d]' % index

            # Since we already filtered out the non-printable types, we can safely dereference the pointer
            if self.Typename.code == gdb.TYPE_CODE_PTR:
                secondStr += " = " + ColorAddress("{}".format(val))
                deRefVal = val.dereference()
                # TODO: for struct, we don't print, just leave the address there, in the future we can print
                # a summary for all AST and IR nodes
                if (deRefVal.type.code != gdb.TYPE_CODE_STRUCT):
                    secondStr += str(deRefVal)
            else:
                secondStr += " = " + str(val)

            if (index == self.elementCount - 1):
                secondStr += prefixStr
            return (firstStr, secondStr)

    def display_hint(self):
        return "array"

    def children(self):
        if self.thisTypePrintable and self.count > 0:
            return self._iterator(self.valobj)
        else:
            return self._iterator(None)

    def to_string(self):
        indent = "    "
        string =  "\n" + indent
        string += ColorVariable("m_count")+ " = " + str(self.count) + "\n"
        string += indent
        string += ColorVariable("m_capacity") + " = " + str(self.m_capacity) + "\n"
        string += indent
        string += ColorVariable("m_buffer") + " (" + ColorType(str(self.buffer.type)) + ")"
        string += " = " + ColorAddress(str(self.buffer)) + "\n"
        string += indent + "\b"
        return string

# Slang::ShortList synthetic provider
class ShortList_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return True

    def num_children(self):
        return self.count.GetValueAsUnsigned(0)

    def get_child_index(self, name):
        return int(name.lstrip("[").rstrip("]"))

    def get_child_at_index(self, index):
        if index >= 0 and index < self.short_count:
            offset = index * self.data_size
            return self.short_buffer.CreateChildAtOffset(
                "[" + str(index) + "]", offset, self.data_type
            )
        elif index >= self.short_count and index < self.num_children():
            offset = (index - self.short_count) * self.data_size
            return self.buffer.CreateChildAtOffset(
                "[" + str(index) + "]", offset, self.data_type
            )
        else:
            return None

    def update(self):
        self.count = self.valobj.GetChildMemberWithName("m_count")
        self.buffer = self.valobj.GetChildMemberWithName("m_buffer")
        self.short_buffer = self.valobj.GetChildMemberWithName("m_shortBuffer")
        self.short_count = self.short_buffer.GetNumChildren()
        self.data_type = self.buffer.GetType().GetPointeeType()
        self.data_size = self.data_type.GetByteSize()

# Slang::VarDeclBase
class DeclDispatcher:
    def __init__(self, val):
        self.val = val
    def to_string(self):
        try:
            import pdb; pdb.set_trace()
            astNoteType = self.val['astNodeType']
            if str(astNoteType) in PRINTER_CLASS_MAP:
                decl_printer = PRINTER_CLASS_MAP[str(astNoteType)](self.val)
                return decl_printer.to_string()

            return ""
        except Exception as e:
            return f"<error reading VarDeclBase: {e}>"

class TypeDispatcher:
    def __init__(self, val):
        self.val = val
    def to_string(self):
        return ""

# Type is a decl ref
class BasicExpressionTypePrinter:
    def __init__(self, val):
        self.val = val
    def to_string(self):
        #((BuiltinTypeModifier*)((Decl*)((DeclRefBase*)((decl->returnType.type)->m_operands.m_buffer[0].values.nodeOperand))->m_operands.m_buffer[0].values.nodeOperand)->modifiers.first).tag
        decl = self.val['type']
        declRefBase = decl['m_operands']['m_buffer'][0]['values']['nodeOperand']
        decl = declRefBase['m_operands']['m_buffer'][0]['values']['nodeOperand']
        modifiers = decl['modifiers']['first']
        tag = modifiers['tag']
        return tag.to_string()

class DeclBasePrinter:
    def __init__(self, val):
        self.val = val
    def to_string(self):
        name_and_loc = self.val['nameAndLoc']
        name = name_and_loc['name']
        astNoteType = self.val['astNodeType']
        name_str = NamePrinter(name).to_string() if name else 'nullptr'

        lines = [
            f"    {ColorVariable('astNodeType')} = {str(astNoteType)}",
            f"    {ColorVariable('name')} = {name_str}",
        ]
        return lines

class FuncDeclPrinter(DeclBasePrinter):
    def __init__(self, val):
        super().__init__(val)  # Call base class constructor
        self.val = val
    def to_string(self):
        try:
            import pdb; pdb.set_trace()
            lines = super().to_string()
            returnType = self.val['returnType']['type']
            returnType_str = ""
            if int(returnType) != 0:
                returnType_str = str(returnType['astNodeType'])

                if returnType_str == "Slang::ASTNodeType::BasicExpressionType":
                    # figure out how to call c++ method from python
                    base_type = gdb.parse_and_eval(f'{returnType}.getBaseType()')
                lines.append(f"    {ColorVariable('returnType')} = {base_type}")
            else:
                lines.append(f"    {ColorVariable('returnType.type')} = nullptr")

            return '\n'.join(lines)

        except Exception as e:
            return f"<error reading FuncDecl: {e}>"

# Global dictionary mapping strings to printer classes
PRINTER_CLASS_MAP = {
    "Slang::ASTNodeType::FuncDecl": FuncDeclPrinter,
    # "Slang::ASTNodeType::VarDecl": DeclBasePrinter,
    # "Slang::ASTNodeType::ParamDecl": DeclBasePrinter,
    # "Slang::ASTNodeType::TypeDefDecl": DeclBasePrinter,
    # "Slang::ASTNodeType::StructDecl": DeclBasePrinter,
    # "Slang::ASTNodeType::ClassDecl": DeclBasePrinter,
}


def register_slang_printers(objfile):
	if objfile == None:
		objfile = gdb.current_objfile()
	gdb.printing.register_pretty_printer(objfile, slang_pretty_printer(), True)
	print("Registered pretty printers for slang core types")


def slang_pretty_printer():
	# add a random numeric suffix to the printer name so we can reload printers during the same session for iteration
    pp = gdb.printing.RegexpCollectionPrettyPrinter("slang")
    pp.add_printer("List", "^Slang::List<.+>$", ListPrinter)
    pp.add_printer("String", "^Slang::String$", StringPrinter)
    pp.add_printer("Name", "^Slang::Name$", NamePrinter)

    # Direct all Decls to DeclBasePrinter
    pp.add_printer("DeclBase", "^Slang::.*Decl$", DeclDispatcher)
    pp.add_printer("DeclBase", "^Slang::.*DeclBase$", DeclDispatcher)
    return pp

register_slang_printers(None)
