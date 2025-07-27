import unittest

from gmx_tracker.configs_parsing import parse_raw
# TODO
raw_input = """# this is a comment, it is ignored, same with empty lines

gmx mdrun -s file.tpr -nt 16|24
gmx mdrun -s file.tpr -update gpu

@gmx mdrun -s file.tpr -nt 8
@gmx mdrun -s file.tpr -nt 8|17
!gmx mdrun -s file.tpr -nt 8
@gmx mdrun -s file.tpr -nt 4|8
# comment in between is legal, they are still treated as paralell 
@gmx mdrun -s file.tpr -nt 4
"""

class ParsingTest(unittest.TestCase):
    def test(self):
        self.assertEqual(parse_raw(raw_input), [[["gmx", "mdrun", "-s", "file.tpr"]]])
