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
        lldb.formatters.Logger.Logger() >> msg




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
       type.code == gdb.TYPE_CODE_PTR or \
       type.code == gdb.TYPE_CODE_STRING or \
       (type.code == gdb.TYPE_CODE_STRUCT and type.name == "Slang::String"):
        return True


# Slang::String synthetic provider
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

# Slang::List synthetic provider
class ListPrinter:
    def __init__(self, valobj):
        self.valobj = valobj
        self.Typename = valobj.type.template_argument(0)
        self.count = valobj['m_count']
        #  print("List_synthetic: %s\n" % self.Typename)
        self.thisTypePrintable = False
        # we don't really want to expand the children if the type is not basic type.
        # The exception is Slang::String, which is a struct but we want to expand it.
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
                self.elementCount = 0
                self.buffer = None
                self.Typename = None
            else:
                self.valobj = valobj
                self.elementCount = self.valobj['m_count']
                self.buffer = self.valobj['m_buffer']
                self.Typename = self.buffer.type
            self.curIndex = 0


        def __iter__(self):
            return self


        def __next__(self):
            if self.elementCount == 0:
                raise StopIteration


            index = self.curIndex
            self.curIndex = self.curIndex + 1
            if self.curIndex > self.elementCount:
                raise StopIteration


            indent = "    "
            prefixStr = '\n' + indent + indent + indent
            firstStr = prefixStr + '[%d]' % index


            val = self.buffer[index]
            secondStr = str(val)


            # Since we already filtered out the non-printable types, we can safely dereference the pointer
            if self.Typename.code == gdb.TYPE_CODE_PTR:
                val = val.dereference()
                secondStr += ": " + str(val)


            return (firstStr, secondStr)

    def children(self):
        if self.thisTypePrintable and self.count > 0:
            return self._iterator(self.valobj)
        else:
            return self._iterator(None)

    def to_string(self):
        #  pass
        count = self.valobj['m_count'];
        m_capacity = self.valobj['m_capacity'];
        buffer = self.valobj['m_buffer'];
        indent = "    "
        string =  "\n" + indent
        string += ColorVariable("m_count")+ " = " + str(count) + "\n"
        string += indent
        string += ColorVariable("m_capacity") + " = " + str(m_capacity) + "\n"
        string += indent
        string += ColorVariable("m_buffer") + " (" + ColorType(str(buffer.type)) + ")"
        string += " = " + ColorAddress(str(buffer)) + "\n"
        string += indent + indent
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
    return pp
