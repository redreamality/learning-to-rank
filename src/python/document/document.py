# This file is part of Lerot.
#
# Lerot is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Lerot.  If not, see <http://www.gnu.org/licenses/>.


class Document(object):

    def __init__(self, docid, doctype=None):
        self.docid = docid
        self.doctype = doctype

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.docid == other.docid)

    def __lt__(self, other):
        return self.docid < other.docid

    def __le__(self, other):
        return self.docid <= other.docid

    def __gt__(self, other):
        return self.docid > other.docid

    def __ge__(self, other):
        return self.docid >= other.docid

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.docid

    def __repr__(self):
        return "Document(id=%d, type=%s)" % (self.docid, self.doctype)

    def __str__(self):
        return self.__repr__()

    def set_type(self, doctype):
        self.doctype = doctype

    def get_type(self):
        return self.doctype

    def get_id(self):
        return self.docid
