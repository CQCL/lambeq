import pytest

from lambeq import BobcatParser, SpacyTokeniser
from lambeq.backend.grammar import Cup, Diagram, Ty, Word
from lambeq.text2diagram.pregroup_tree import PregroupTreeNode
from lambeq.text2diagram.pregroup_tree_converter import (
    diagram2tree, generate_tree, tree2diagram
)


tokeniser = SpacyTokeniser()
bobcat_parser = BobcatParser(verbose='suppress')
n, s, p = map(Ty, 'nsp')

s1 = tokeniser.tokenise_sentence(
    "Last year's figures include a one-time loss of $12 million for restructuring and unusual items"
)
s2 = tokeniser.tokenise_sentence('Do your homework')
s3 = tokeniser.tokenise_sentence('I like but Mary dislikes reading')
s12 = tokeniser.tokenise_sentence("i don't like it")
s13 = tokeniser.tokenise_sentence("i haven't seen it")

s1_diag = bobcat_parser.sentence2diagram(s1, tokenised=True)
s2_diag = bobcat_parser.sentence2diagram(s2, tokenised=True)
s3_diag = bobcat_parser.sentence2diagram(s3, tokenised=True)
s4_diag = Diagram.create_pregroup_diagram(
    words=[
        Word('Do', n.r @ s @ n.l),
        Word('that', n @ n.l),
    ],
    morphisms=[
        (Cup, 2, 3)
    ]
)
s5_diag = Diagram.create_pregroup_diagram(
    words=[
        Word('a', n.l @ n.l),
        Word('b', n @ n.l),
        Word('c', n @ n),
    ],
    morphisms=[
        (Cup, 1, 2),
        (Cup, 3, 4),
        (Cup, 0, 5),
    ]
)
s6_diag = Diagram.create_pregroup_diagram(
    words=[
        Word('parent', s @ n @ n.l @ n.l @ n.l),
        Word('x', n @ n.l),
        Word('Mary', n @ n @ n @ n.l @ n.l),
        Word('y', n @ n @ n.r),
    ],
    morphisms=[
        (Cup, 4, 5),
        (Cup, 6, 7),
        (Cup, 11, 12),
        (Cup, 3, 8),
        (Cup, 10, 13),
        (Cup, 2, 9),
        (Cup, 1, 14),
    ]
)
s8_diag = bobcat_parser.sentence2diagram(
    'The book Alice said Bob liked but she did not was that one',
    tokenised=False
)
s9_diag = Diagram.create_pregroup_diagram(
    words=[
        Word('0', n.l @ n.l),
        Word('1', n @ n.l),
        Word('2', n @ n @ n.l),
        Word('3', n @ n.l @ n.l),
        Word('4', n @ n.l),
        Word('5', n @ n @ s),
    ],
    morphisms=[
        (Cup, 1, 2),
        (Cup, 3, 4),
        (Cup, 6, 7),
        (Cup, 9, 10),
        (Cup, 11, 12),
        (Cup, 0, 5),
        (Cup, 8, 13),
    ]
)
s10_diag = Diagram.create_pregroup_diagram(
    words=[
        Word('0', s @ n.l @ n.l),
        Word('1', n @ n.l),
        Word('2', n @ n @ n.l @ n.l),
        Word('3', n @ n.l),
        Word('4', n @ n),
    ],
    morphisms=[
        (Cup, 2, 3),
        (Cup, 4, 5),
        (Cup, 8, 9),
        (Cup, 10, 11),
        (Cup, 1, 6),
        (Cup, 7, 12),
    ]
)
s11_diag = bobcat_parser.sentence2diagram(
    'When an event puts Errol in danger and the case in jeopardy',
    tokenised=False
)
s12_diag = bobcat_parser.sentence2diagram(s12, tokenised=True)
s13_diag = bobcat_parser.sentence2diagram(s13, tokenised=True)

t1_n1 = PregroupTreeNode(word='year', typ=n, ind=1)
t1_n3 = PregroupTreeNode(word='figures', typ=n, ind=3)
t1_n8 = PregroupTreeNode(word='loss', typ=n, ind=8)
t1_n12 = PregroupTreeNode(word='million', typ=n, ind=12)
t1_n14 = PregroupTreeNode(word='restructuring', typ=n, ind=14)
t1_n17 = PregroupTreeNode(word='items', typ=n, ind=17)
t1_n0 = PregroupTreeNode(word='Last', typ=n, ind=0, children=[t1_n1])
t1_n2 = PregroupTreeNode(word="'s", typ=n, ind=2, children=[t1_n0, t1_n3])
t1_n7 = PregroupTreeNode(word='time', typ=n, ind=7, children=[t1_n8])
t1_n6 = PregroupTreeNode(word='one', typ=n, ind=6, children=[t1_n7])
t1_n5 = PregroupTreeNode(word='a', typ=n, ind=5, children=[t1_n6])
t1_n11 = PregroupTreeNode(word='12', typ=n, ind=11, children=[t1_n12])
t1_n10 = PregroupTreeNode(word='$', typ=n, ind=10, children=[t1_n11])
t1_n16 = PregroupTreeNode(word='unusual', typ=n, ind=16, children=[t1_n17])
t1_n15 = PregroupTreeNode(word='and', typ=n, ind=15, children=[t1_n14, t1_n16])
t1_n9 = PregroupTreeNode(word='of', typ=n, ind=9, children=[t1_n5, t1_n10])
t1_n13 = PregroupTreeNode(word='for', typ=n, ind=13, children=[t1_n9, t1_n15])
t1_n4 = PregroupTreeNode(word='include', typ=s, ind=4, children=[t1_n2, t1_n13])
t1 = t1_n4

t2_n2 = PregroupTreeNode(word='homework', typ=n, ind=2)
t2_n1 = PregroupTreeNode(word='your', typ=n, ind=1, children=[t2_n2])
t2_n0 = PregroupTreeNode(word='Do', typ=n.r @ s, ind=0, children=[t2_n1])
t2 = t2_n0

t3_n0 = PregroupTreeNode(word='I', typ=n, ind=0)
t3_n1 = PregroupTreeNode(word='like', typ=n.r @ s @ n.l, ind=1)
t3_n5 = PregroupTreeNode(word='reading', typ=n, ind=5)
t3_n4 = PregroupTreeNode(word='dislikes', typ=n.r @ s, ind=4)
t3_n4_2 = PregroupTreeNode(word='dislikes', typ=n.l, ind=4)
t3_n3 = PregroupTreeNode(word='Mary', typ=n.r @ s, ind=3, children=[t3_n4])
t3_n2 = PregroupTreeNode(word='but', typ=s, ind=2,
                         children=[t3_n0, t3_n1, t3_n3, t3_n4_2, t3_n5])
t3 = t3_n2

t5_n2 = PregroupTreeNode(word='c', typ=n, ind=2)
t5_n1 = PregroupTreeNode(word='b', typ=n, ind=1, children=[t5_n2])
t5_n2_2 = PregroupTreeNode(word='c', typ=n, ind=2)
t5_n0 = PregroupTreeNode(word='a', typ=Ty(), ind=0,
                         children=[t5_n1, t5_n2_2])
t5 = t5_n0

t6_n2 = PregroupTreeNode(word='Mary', typ=n.l @ n.l, ind=2)
t6_n1 = PregroupTreeNode(word='x', typ=n.l, ind=1)
t6_n1_2 = PregroupTreeNode(word='x', typ=n, ind=1)
t6_n2_2 = PregroupTreeNode(word='Mary', typ=n @ n, ind=2, children=[t6_n1])
t6_n3 = PregroupTreeNode(word='y', typ=n.r, ind=3, children=[t6_n2])
t6_n0 = PregroupTreeNode(word='parent', typ=s, ind=0,
                         children=[t6_n1_2, t6_n2_2, t6_n3])
t6 = t6_n0

t7_n2 = PregroupTreeNode(word='Mary', typ=n, ind=2)
t7_n3 = PregroupTreeNode(word='y', typ=n @ n, ind=3)
t7_n3_2 = PregroupTreeNode(word='y', typ=n.r, ind=3)
t7_n2_2 = PregroupTreeNode(word='Mary', typ=n @ n, ind=2, children=[t7_n3])
t7_n1 = PregroupTreeNode(word='x', typ=n, ind=1, children=[t7_n2])
t7_n0 = PregroupTreeNode(word='parent', typ=s, ind=0,
                         children=[t7_n1, t7_n2_2, t7_n3_2])
t7 = t7_n0

t8_n1 = PregroupTreeNode(word='book', typ=n, ind=1)
t8_n2 = PregroupTreeNode(word='Alice', typ=n, ind=2)
t8_n4 = PregroupTreeNode(word='Bob', typ=n, ind=4)
t8_n5 = PregroupTreeNode(word='liked', typ=n.r @ s @ n.l, ind=5)
t8_n8 = PregroupTreeNode(word='did', typ=n.r @ s, ind=8)
t8_n9 = PregroupTreeNode(word='not', typ=n.r @ s, ind=9, children=[t8_n8])
t8_n7 = PregroupTreeNode(word='she', typ=n.r @ s, ind=7, children=[t8_n9])
t8_n12 = PregroupTreeNode(word='one', typ=n, ind=12)
t8_n0 = PregroupTreeNode(word='The', typ=n, ind=0, children=[t8_n1])
t8_n11 = PregroupTreeNode(word='that', typ=n, ind=11, children=[t8_n12])
t8_n6 = PregroupTreeNode(word='but', typ=s, ind=6,
                         children=[t8_n0, t8_n4, t8_n5, t8_n7])
t8_n3 = PregroupTreeNode(word='said', typ=n, ind=3, children=[t8_n2, t8_n6])
t8_n10 = PregroupTreeNode(word='was', typ=s, ind=10, children=[t8_n3, t8_n11])
t8_no_cycle = t8_n10

t9_n1 = PregroupTreeNode(word='1', typ=n, ind=1)
t9_n4 = PregroupTreeNode(word='4', typ=n, ind=4)
t9_n0 = PregroupTreeNode(word='0', typ=n.l, ind=0, children=[t9_n1])
t9_n2 = PregroupTreeNode(word='2', typ=n.l, ind=2, children=[t9_n0])
t9_n3 = PregroupTreeNode(word='3', typ=n.l, ind=3, children=[t9_n2, t9_n4])
t9_n5 = PregroupTreeNode(word='5', typ=s, ind=5, children=[t9_n3])
t9_no_cycle = t9_n5

t10_n4 = PregroupTreeNode(word='4', typ=n, ind=4)
t10_n3 = PregroupTreeNode(word='3', typ=n, ind=3, children=[t10_n4])
t10_n2 = PregroupTreeNode(word='2', typ=n, ind=2, children=[t10_n3])
t10_n1 = PregroupTreeNode(word='1', typ=n, ind=1, children=[t10_n2])
t10_n0 = PregroupTreeNode(word='0', typ=s, ind=0, children=[t10_n1])
t10_no_cycle = t10_n0

t11_n11 = PregroupTreeNode(word='jeopardy', typ=n, ind=11)
t11_n9 = PregroupTreeNode(word='case', typ=n, ind=9)
t11_n6 = PregroupTreeNode(word='danger', typ=n, ind=6)
t11_n4 = PregroupTreeNode(word='Errol', typ=n, ind=4)
t11_n3 = PregroupTreeNode(word='puts', typ=n.r @ s @ p.l @ n.l, ind=3)
t11_n2 = PregroupTreeNode(word='event', typ=n, ind=2)
t11_n10 = PregroupTreeNode(word='in', typ=p, ind=10, children=[t11_n11])
t11_n8 = PregroupTreeNode(word='the', typ=n, ind=8, children=[t11_n9])
t11_n5 = PregroupTreeNode(word='in', typ=p, ind=5, children=[t11_n6])
t11_n1 = PregroupTreeNode(word='an', typ=n, ind=1, children=[t11_n2])
t11_n7 = PregroupTreeNode(word='and', typ=s, ind=7,
                          children=[t11_n1, t11_n3, t11_n4,
                                    t11_n5, t11_n8, t11_n10])
t11_n0 = PregroupTreeNode(word='When', typ=s, ind=0, children=[t11_n7])
t11_no_cycle = t11_n0

t12_n4 = PregroupTreeNode(word='it', typ=n, ind=4)
t12_n0 = PregroupTreeNode(word='i', typ=n, ind=0)
t12_n3 = PregroupTreeNode(word='like', typ=n.r @ s, ind=3, children=[t12_n4])
t12_n1 = PregroupTreeNode(word='do', typ=n.r @ s, ind=1, children=[t12_n3])
t12_n2 = PregroupTreeNode(word="n't", typ=s, ind=2, children=[t12_n0, t12_n1])
t12 = t12_n2

t13_n4 = PregroupTreeNode(word='it', typ=n, ind=4)
t13_n0 = PregroupTreeNode(word='i', typ=n, ind=0)
t13_n3 = PregroupTreeNode(word='seen', typ=n.r @ s, ind=3, children=[t13_n4])
t13_n1 = PregroupTreeNode(word='have', typ=n.r @ s, ind=1, children=[t13_n3])
t13_n2 = PregroupTreeNode(word="n't", typ=s, ind=2, children=[t13_n0, t13_n1])
t13 = t13_n2


def test_diagram2tree():
    s1_tree = diagram2tree(s1_diag)
    s1_tree.draw()
    t1.draw()
    assert s1_tree == t1

    # Root node has compound type i.e. multiple free string
    s2_tree = diagram2tree(s2_diag)
    assert s2_tree == t2

    # Cycles are valid trees - we represent cycles by duplicating
    # one of the nodes in the cycle
    s3_tree = diagram2tree(s3_diag)
    assert s3_tree == t3

    # Multiple free strings but on separate nodes
    with pytest.raises(ValueError):
        s4_tree = diagram2tree(s4_diag)

    s5_tree = diagram2tree(s5_diag)
    assert s5_tree == t5


def test_diagram2tree_no_cycles():
    s8_tree = diagram2tree(s8_diag, break_cycles=True)
    s8_tree.draw()
    t8_no_cycle.draw()
    assert s8_tree == t8_no_cycle

    s9_tree = diagram2tree(s9_diag, break_cycles=True)
    assert s9_tree == t9_no_cycle

    s10_tree = diagram2tree(s10_diag, break_cycles=True)
    s10_tree.draw()
    t10_no_cycle.draw()
    assert s10_tree == t10_no_cycle

    s11_tree = diagram2tree(s11_diag, break_cycles=True)
    s11_tree.draw()
    t11_no_cycle.draw()
    assert s11_tree == t11_no_cycle


def test_tree2diagram():
    assert tree2diagram(t1, t1.get_words()).pregroup_normal_form() == s1_diag.pregroup_normal_form()
    assert tree2diagram(t2, t2.get_words()).pregroup_normal_form() == s2_diag.pregroup_normal_form()
    assert tree2diagram(t3, t3.get_words()).pregroup_normal_form() == s3_diag.pregroup_normal_form()
    # These two are the same diagrams but slightly different tree encodings
    assert tree2diagram(t5, t5.get_words()).pregroup_normal_form() == s5_diag.pregroup_normal_form()
    assert tree2diagram(t6, t6.get_words()).pregroup_normal_form() == s6_diag.pregroup_normal_form()
    assert tree2diagram(t7, t7.get_words()).pregroup_normal_form() == s6_diag.pregroup_normal_form()
    assert tree2diagram(t12, t12.get_words()).pregroup_normal_form() == s12_diag.pregroup_normal_form()
    assert tree2diagram(t13, t13.get_words()).pregroup_normal_form() == s13_diag.pregroup_normal_form()

def test_generate_tree():
    # valid diagram
    tokens = ['In', 'Thailand', 'for', 'example', 'the', 'government', "'s", 'Board', 'of', 'Investment', 'approved', '$', '705.6', 'million', 'of', 'Japanese', 'investment', 'in', '1988', '10', 'times', 'the', 'U.S.', 'investment', 'figure', 'for', 'the', 'year']
    types = ['s', 'n', 's', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n.r @ s', 'n', 'n', 'n', 'n', 'n', 'n', 'n.r @ s', 'n', 'n', 'n @ n.l', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
    types = [[t] for t in types]
    parents = [-1, 0, 0, 2, 6, 4, 8, 6, 25, 8, 17, 14, 11, 12, 10, 14, 15, 25, 17, 25, 19, 19, 21, 22, 23, 2, 25, 26]
    parents = [[p] for p in parents]
    root_nodes, nodes = generate_tree(tokens, types, parents)
    assert len(root_nodes) == 1
    root_nodes[0].draw()
    assert nodes[3][0].parent == nodes[2][0]
    assert nodes[4][0].parent == nodes[6][0]

    # multiple root nodes
    tokens = ['In', 'Thailand', 'for', 'example']
    types = [['s'], ['n'], ['s'], ['n']]
    parents = [[-1], [0], [-1], [2]]
    root_nodes, _ = generate_tree(tokens, types, parents)
    assert len(root_nodes) == 2

    # false root (node parent is itself)
    tokens = ['In', 'Thailand', 'for', 'example']
    types = [['s'], ['n'], ['s'], ['n']]
    parents = [[-1], [0], [2], [2]]
    with pytest.raises(ValueError):
        root_nodes, _ = generate_tree(tokens, types, parents)

    # cycle - break cycle and assign one of the cycle nodes parent to root
    tokens = ['In', 'Thailand', 'for', 'example', 'for']
    types = [['s'], ['n'], ['s'], ['n'], ['s']]
    parents = [[-1], [0], [4], [2], [2]]
    root_nodes, _ = generate_tree(tokens, types, parents)
    # assert len(root_nodes) == 2
    for root in root_nodes:
        root.draw()

    # valid cycle
    tokens = ['I', 'like', 'but', 'Mary', 'dislikes', 'reading']
    types = [['n'], ['n.r @ s @ n.l'], ['s'], ['n.r @ s', 's.l @ n'], ['n.l'], ['n']]
    parents = [[2], [2], [-1], [2, 4], [2], [2]]
    root_nodes, _ = generate_tree(tokens, types, parents)
    assert len(root_nodes) == 1
    root_nodes[0].draw()

    # valid cycle (+3 nodes in cycle)
    tokens = ['I', 'like', 'but', 'Mary', 'Anne', 'dislikes', 'reading']
    types = [['n'], ['n.r @ s @ n.l'], ['s'], ['n.r @ s'], ['s.l @ n', 'n'], ['n.l'], ['n']]
    parents = [[2], [2], [-1], [2], [3, 5], [2], [2]]
    root_nodes, _ = generate_tree(tokens, types, parents)
    assert len(root_nodes) == 1
    root_nodes[0].draw()

    # More cycles, no free strings
    # Counterclockwise (ccw) 4-cycle with "twist"
    root_nodes, _ = generate_tree(['a', 'b', 'c', 'd'],
                                  [['n'], ['n'], ['n'], ['n']],
                                  [[1], [3], [0], [2]])
    assert len(root_nodes) == 1

    # Clockwise (cw) 4-cycle with "twist"
    root_nodes, _ = generate_tree(['a', 'b', 'c', 'd'],
                                  [['n'], ['n'], ['n'], ['n']],
                                  [[2], [0], [3], [1]])
    assert len(root_nodes) == 1

    # CCW 3-cycle
    root_nodes, _ = generate_tree(['a', 'b', 'c'],
                                  [['n'], ['n'], ['n']],
                                  [[2], [0], [1]])
    assert len(root_nodes) == 1

    # CW 3-cycle
    root_nodes, _ = generate_tree(['a', 'b', 'c'],
                                  [['n'], ['n'], ['n']],
                                  [[1], [2], [0]])
    assert len(root_nodes) == 1

    # Two cycles
    root_nodes, _ = generate_tree(['a', 'b', 'c', 'd', 'e'],
                                  [['n'], ['n'], ['n'], ['n'], ['n']],
                                  [[1], [2], [0], [4], [3]])
    assert len(root_nodes) == 2
