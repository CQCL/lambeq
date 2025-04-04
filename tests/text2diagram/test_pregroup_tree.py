from lambeq.backend.grammar import Ty
from lambeq.text2diagram.pregroup_tree import PregroupTreeNode


n, s = map(Ty, 'ns')

t1_n1 = PregroupTreeNode(word='year', typ=n, ind=1)
t1_n3 = PregroupTreeNode(word='figures', typ=n, ind=3)
t1_n8 = PregroupTreeNode(word='loss', typ=n, ind=8)
t1_n12 = PregroupTreeNode(word='million', typ=n, ind=12)
t1_n14 = PregroupTreeNode(word='restructuring', typ=n, ind=14)
t1_n17 = PregroupTreeNode(word='items', typ=n, ind=17)
t1_n0 = PregroupTreeNode(word='Last', typ=n, ind=0,
                         children=[t1_n1])
t1_n2 = PregroupTreeNode(word="'s", typ=n, ind=2,
                         children=[t1_n0, t1_n3])
t1_n7 = PregroupTreeNode(word='time', typ=n, ind=7,
                         children=[t1_n8])
t1_n6 = PregroupTreeNode(word='one', typ=n, ind=6,
                         children=[t1_n7])
t1_n5 = PregroupTreeNode(word='a', typ=n, ind=5,
                         children=[t1_n6])
t1_n11 = PregroupTreeNode(word='12', typ=n, ind=11,
                          children=[t1_n12])
t1_n10 = PregroupTreeNode(word='$', typ=n, ind=10,
                          children=[t1_n11])
t1_n16 = PregroupTreeNode(word='unusual', typ=n, ind=16,
                          children=[t1_n17])
t1_n15 = PregroupTreeNode(word='and', typ=n, ind=15,
                          children=[t1_n14, t1_n16])
t1_n9 = PregroupTreeNode(word='of', typ=n, ind=9,
                         children=[t1_n5, t1_n10])
t1_n13 = PregroupTreeNode(word='for', typ=n, ind=13,
                          children=[t1_n9, t1_n15])
t1_n4 = PregroupTreeNode(word='include', typ=s, ind=4,
                         children=[t1_n2, t1_n13])
t1 = t1_n4
t1_types = [['n']] * 18
t1_types[4] = ['s']
t1_parents = [[2], [0], [4], [2], [-1], [9], [5], [6], [7], [13],
              [9], [10], [11], [4], [15], [13], [15], [16]]

t2_n1 = PregroupTreeNode(word='was', typ=n.r @ s, ind=1)
t2_n2 = PregroupTreeNode(word='not', typ=n.r @ s, ind=2,
                         children=[t2_n1])
t2_n0 = PregroupTreeNode(word='root', typ=s, ind=0, children=[t2_n2])
t2 = t2_n0

t3_n1 = PregroupTreeNode(word='was', typ=n.r @ s, ind=4)
t3_n2 = PregroupTreeNode(word='not', typ=n.r @ s, ind=2,
                         children=[t3_n1])
t3_n0 = PregroupTreeNode(word='root', typ=s, ind=0, children=[t3_n2])
t3 = t3_n0

t4_n4 = PregroupTreeNode(word='4', typ=n, ind=4)
t4_n4_2 = PregroupTreeNode(word='4', typ=n.r, ind=4)
t4_n3 = PregroupTreeNode(word='3', typ=n, ind=3, children=[t4_n4])
t4_n2 = PregroupTreeNode(word='2', typ=n, ind=2,
                         children=[t4_n3, t4_n4_2])
t4_n2_2 = PregroupTreeNode(word='2', typ=n.r, ind=2)
t4_n1 = PregroupTreeNode(word='1', typ=n, ind=1, children=[t4_n2])
t4_n0 = PregroupTreeNode(word='0', typ=s, ind=0,
                         children=[t4_n1, t4_n2_2])
t4 = t4_n0

t5_n2 = PregroupTreeNode(word='and', typ=n.r @ s @ n.r.r.r @ s.r.r,
                         ind=2)
t5_n1 = PregroupTreeNode(word='an', typ=n, ind=1)
t5_n2_2 = PregroupTreeNode(word='and', typ=s, ind=2,
                           children=[t5_n1, t5_n2])
t5_n0 = PregroupTreeNode(word='when', typ=s, ind=0, children=[t5_n2_2])
t5 = t5_n0


def test_eq():
    t1_n1 = PregroupTreeNode(word='t1', typ=Ty(), ind=1)
    t1_n2 = PregroupTreeNode(word='t2', typ=Ty(), ind=2)
    t1_n0 = PregroupTreeNode(word='t0', typ=Ty(), ind=0,
                             children=[t1_n1, t1_n2])
    t1 = t1_n0

    t2_n1 = PregroupTreeNode(word='t1', typ=Ty(), ind=1)
    t2_n2 = PregroupTreeNode(word='t2', typ=Ty(), ind=2)
    t2_n0 = PregroupTreeNode(word='t0', typ=Ty(), ind=0,
                             children=[t2_n2, t2_n1])
    t2 = t2_n0
    assert t1 == t2


def test_get_nodes():
    assert t1.get_nodes() == [
        [t1_n0],
        [t1_n1],
        [t1_n2],
        [t1_n3],
        [t1_n4],
        [t1_n5],
        [t1_n6],
        [t1_n7],
        [t1_n8],
        [t1_n9],
        [t1_n10],
        [t1_n11],
        [t1_n12],
        [t1_n13],
        [t1_n14],
        [t1_n15],
        [t1_n16],
        [t1_n17],
    ]
    assert t1_n13.get_nodes() == [
        [t1_n5],
        [t1_n6],
        [t1_n7],
        [t1_n8],
        [t1_n9],
        [t1_n10],
        [t1_n11],
        [t1_n12],
        [t1_n13],
        [t1_n14],
        [t1_n15],
        [t1_n16],
        [t1_n17],
    ]

    # With cycles (as duplicate nodes)
    assert t4.get_nodes() == [
        [t4_n0],
        [t4_n1],
        [t4_n2, t4_n2_2],
        [t4_n3],
        [t4_n4, t4_n4_2],
    ]


def test_get_types():
    assert t1.get_types() == t1_types
    assert t1_n13.get_types() == [['n']] * 13


def test_get_parents():
    assert t1.get_parents() == t1_parents
    assert t1_n13.get_parents() == [
        [9], [5], [6], [7], [13], [9], [10], [11], [4], [15],
        [13], [15], [16]
    ]


def test_get_word_indices():
    assert t1.get_word_indices() == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    ]
    assert t1_n13.get_word_indices() == [
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    ]


def test_get_words():
    assert t1.get_words() == [
        'Last', 'year', '\'s', 'figures', 'include',    # noqa: Q003
        'a', 'one', 'time', 'loss', 'of',
        '$', '12', 'million', 'for', 'restructuring',
        'and', 'unusual', 'items',
    ]


def test_parent_attr():
    assert t1.parent is None
    assert t1_n2.parent == t1_n4
    assert t1_n13.parent == t1_n4
    assert t1_n14.parent == t1_n15
    assert t1_n1.parent == t1_n0


def test_get_root():
    assert t1.get_root() == t1
    assert t1_n15.get_root() == t1
    assert t1_n11.get_root() == t1
    assert t1_n7.get_root() == t1
    assert t1_n3.get_root() == t1


def test_get_depth():
    # No arguments - compute depth relative to root
    assert t1_n4.get_depth() == 0
    assert t1_n5.get_depth() == 3
    assert t1_n8.get_depth() == 6

    # Get depth in a subtree
    assert t1_n7.get_depth(t1_n9) == 3
    assert t1_n14.get_depth(t1_n15) == 1
    assert t1_n17.get_depth(t1_n15) == 2

    # Node not in tree specified by the root
    new_node = PregroupTreeNode(word='test', typ=n, ind=7)
    assert new_node.get_depth(t1) == -1


def test_repr():
    assert str(t1) == 'include_4 (s)'


def test_tree_repr():
    assert t1._tree_repr == '\n'.join([
        'include_4 (s)',
        '├ \'s_2 (n)',  # noqa: Q003
        '│ ├ Last_0 (n)',
        '│ │ └ year_1 (n)',
        '│ └ figures_3 (n)',
        '└ for_13 (n)',
        '  ├ of_9 (n)',
        '  │ ├ a_5 (n)',
        '  │ │ └ one_6 (n)',
        '  │ │   └ time_7 (n)',
        '  │ │     └ loss_8 (n)',
        '  │ └ $_10 (n)',
        '  │   └ 12_11 (n)',
        '  │     └ million_12 (n)',
        '  └ and_15 (n)',
        '    ├ restructuring_14 (n)',
        '    └ unusual_16 (n)',
        '      └ items_17 (n)',
    ])


def test_merge():
    # Multiple children - do nothing
    t1_n13.merge()
    assert t1_n13.word == 'for'
    assert t1_n13.typ == n
    assert t1_n13.ind == 13
    assert len(t1_n13.children) == 2
    assert all([c.parent == t1_n13 for c in t1_n13.children])

    # One level
    t1_n16.merge()
    assert t1_n16.word == 'unusual items'
    assert t1_n16.typ == n
    assert t1_n16.ind == 16
    assert not t1_n16.children
    assert t1_n17.parent is None

    # Multiple levels - not required
    t1_n10.merge()
    assert t1_n10.word == '$ 12'
    assert t1_n10.typ == n
    assert t1_n10.ind == 10
    assert t1_n10.children == t1_n11.children
    assert t1_n11.parent is None
    assert all([c.parent == t1_n10 for c in t1_n11.children])

    t1_n5.merge()
    assert t1_n5.word == 'a one'
    assert t1_n5.typ == n
    assert t1_n5.ind == 5
    assert t1_n5.children == t1_n6.children
    assert t1_n6.parent is None
    assert all([c.parent == t1_n5 for c in t1_n16.children])

    # Preserve order of words
    t2_n2.merge()
    assert t2_n2.word == 'was not'
    assert t2_n2.typ == n.r @ s
    assert t2_n2.ind == 1
    assert not t2_n2.children

    # Only process consecutive tokens
    t3_n2.merge()
    assert t3_n2.word == 'not'
    assert t3_n2.typ == n.r @ s
    assert t3_n2.ind == 2
    assert len(t3_n2.children) == 1


def test_remove_self_cycles():
    t5.remove_self_cycles()
    assert t5_n2.parent is None
    assert t5_n2_2.children == [t5_n1]
