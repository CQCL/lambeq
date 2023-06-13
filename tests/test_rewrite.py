import pytest

from discopy import Word
from discopy.rigid import Box, Cap, Cup, Diagram, Id, Ob, Spider, Swap, Ty, cups

from lambeq import (AtomicType, Rewriter, CoordinationRewriteRule,
                    CurryRewriteRule, SimpleRewriteRule,
                    UNKRewriteRule)

from lambeq.rewrite.base import HandleUnknownWords, dataset_to_words, remove_words

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_initialisation():
    assert (Rewriter().rules == Rewriter([Rewriter._available_rules[rule]
            for rule in Rewriter._default_rules]).rules)

    assert all([rule in Rewriter.available_rules() for rule in Rewriter._default_rules])
    with pytest.raises(ValueError):
        Rewriter(['nonexistent rule'])


def test_custom_rewriter():
    placeholder = SimpleRewriteRule.placeholder(S)
    not_box = Box('NOT', S, S)
    not_rewriter = SimpleRewriteRule(
        cod=S, template=placeholder >> not_box)

    diagram = Word('I think', S)
    assert not_rewriter(diagram) == diagram >> not_box


def test_auxiliary():
    cod = (N >> S) << (N >> S)
    we = Word('we', N)
    go = Word('go', N >> S)
    cups = Cup(N, N.r) @ Id(S) @ Diagram.cups((N >> S).l, N >> S)

    diagram = (we @ Word('will', cod) @ go) >> cups
    expected_diagram = (we @ Diagram.caps(cod[:2], cod[2:]) @ go) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['auxiliary'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_connector():
    left_words = Word('I', N) @ Word('hope', N >> S << S)
    right_words = Word('this', N) @ Word('succeeds', N >> S)
    cups = (Cup(N, N.r) @ Id(S) @ Cup(S.l, S) @
            Diagram.cups((N >> S).l, N >> S))

    diagram = (left_words @ Word('that', S << S) @ right_words) >> cups
    expected_diagram = (left_words @ Cap(S, S.l) @ right_words) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['connector'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_determiner():
    book = Word('book', N)
    cups = Id(N) @ Cup(N.l, N)

    diagram = (Word('the', N << N) @ book) >> cups
    expected_diagram = (Cap(N, N.l) @ book) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['determiner'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_postadverb():
    cod = (N >> S) >> (N >> S)
    vp = Word('we', N) @ Word('go', N >> S)
    cups = Diagram.cups(cod[:3].l, cod[:3]) @ Id(S)

    diagram = (vp @ Word('quickly', cod)) >> cups
    expected_diagram = (vp @ (Word('quickly', S >> S) >>
                              Id(S.r) @ Cap(N.r.r, N.r) @ Id(S))) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['postadverb'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_preadverb():
    we = Word('we', N)
    go = Word('go', N >> S)
    cups = Cup(N, N.r) @ Id(S) @ Diagram.cups((N >> S).l, N >> S)

    diagram = (we @ Word('quickly', (N >> S) << (N >> S)) @ go) >> cups
    expected_diagram = (we @ (Cap(N.r, N) >>
                              Id(N.r) @ Word('quickly', S << S) @ Id(N)) @
                        go) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['preadverb'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_prepositional_phrase():
    cod = (N >> S) >> (N >> S << N)
    vp = Word('I', N) @ Word('go', N >> S)
    bed = Word('bed', N)
    cups = Diagram.cups(cod[:3].l, cod[:3]) @ Id(S) @ Cup(N.l, N)

    diagram = (vp @ Word('to', cod) @ bed) >> cups
    expected_diagram = (vp @ (Word('to', S >> S << N) >>
                              Id(S.r) @ Cap(N.r.r, N.r) @ Id(S << N)) @ bed >>
                        cups)

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['prepositional_phrase'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_rel_pronoun():
    cows = Word('cows', N)
    that_subj = Word('that', N.r @ N @ S.l @ N)
    that_obj = Word('that', N.r @ N @ N.l.l @ S.l)
    eat = Word('eat', N >> S << N)
    grass = Word('grass', N)

    rewriter = Rewriter(['subject_rel_pronoun', 'object_rel_pronoun'])

    diagram_subj = Id().tensor(cows, that_subj, eat, grass)
    diagram_subj >>= Cup(N, N.r) @ Id(N) @ cups(S.l @ N, N.r @ S) @ Cup(N.l, N)

    expected_diagram_subj = Diagram(
            dom=Ty(), cod=N,
            boxes=[cows, Spider(1, 2, N), Spider(0, 1, S.l), eat, Cup(N, N.r),
                   Cup(S.l, S), grass, Cup(N.l, N)],
            offsets=[0, 0, 1, 3, 2, 1, 2, 1])

    assert rewriter(diagram_subj).normal_form() == expected_diagram_subj

    diagram_obj = Id().tensor(grass, that_obj, cows, eat)
    diagram_obj >>= Cup(N, N.r) @ Id(N) @ Id(N.l.l @ S.l) @ Cup(N, N.r) @ Id(S @ N.l)
    diagram_obj >>= Id(N) @ cups(N.l.l @ S.l, S @ N.l)

    expected_diagram_obj = Diagram(
            dom=Ty(), cod=N,
            boxes=[grass, Spider(1, 2, N), Cap(N.l, N.l.l), Swap(N.l, N.l.l),
                   Spider(0, 1, S.l), cows, eat, Cup(N, N.r), Cup(S.l, S),
                   Cup(N.l.l, N.l), Cup(N.l, N)],
            offsets=[0, 0, 1, 1, 2, 3, 4, 3, 2, 1, 1])

    assert rewriter(diagram_obj).normal_form() == expected_diagram_obj


def test_coordination():
    eggs = Word('eggs', N)
    ham = Word('ham', N)

    words = eggs @ Word('and', N >> N << N) @ ham
    cups = Cup(N, N.r) @ Id(N) @ Cup(N.l, N)

    rewriter = Rewriter([CoordinationRewriteRule()])
    diagram = words >> cups
    expected_diagram = eggs @ ham >> Spider(2, 1, N)

    assert rewriter(diagram).normal_form() == expected_diagram


def test_curry_functor():
    n, s = map(Ty, 'ns')
    diagram = (
        Word('I', n) @ Word('see', n.r @ s @ n.l) @
        Word('the', n @ n.l) @ Word('truth', n)).cup(5, 6).cup(3, 4).cup(0, 1)
    expected = (Word('I', n) @ Word('truth', n) >> Id(n) @ Box('the', n, n)
                >> Box('see', n @ n, s))

    rewriter = Rewriter([CurryRewriteRule()])
    assert rewriter(diagram).normal_form() == expected
    
    
def test_unk_rewrite_rule():
    diagram = Diagram(dom=Ty(), cod=Ty('s'),
            boxes=[Word('I', Ty('n')),
                   Word('see', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
                   Cup(Ty('n'), Ty(Ob('n', z=1))),
                   Word('the', Ty('n', Ob('n', z=-1))),
                   Cup(Ty(Ob('n', z=-1)), Ty('n')), Word('photo', Ty('n')),
                   Cup(Ty(Ob('n', z=-1)), Ty('n'))],
            offsets=[0, 1, 0, 2, 1, 2, 1])

    expected_diagram = Diagram(dom=Ty(), cod=Ty('s'),
            boxes=[Word('I', Ty('n')),
                   Word('see', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
                   Cup(Ty('n'), Ty(Ob('n', z=1))),
                   Word('the', Ty('n', Ob('n', z=-1))),
                   Cup(Ty(Ob('n', z=-1)), Ty('n')), Word('UNK', Ty('n')),
                   Cup(Ty(Ob('n', z=-1)), Ty('n'))],
            offsets=[0, 1, 0, 2, 1, 2, 1])

    unknown_words = ['photo']
    rule = UNKRewriteRule(words = unknown_words)

    rewritten_diagram = Rewriter([rule])(diagram)

    assert rewritten_diagram == expected_diagram


def test_HandleUnknownWords():

    # Train data sentences
    train_data = ['she was happy to take a photo .',
                  'he said that he was lonely .',
                  'depression is caused by holding in anger .',
                  'he suffered agonies of remorse .',
                  'he helps people every day .',
                  'man cooks a lovely dinner .',
                  'she recalled her happy dream with him .',
                  'a happy heart makes a blooming visage .',
                  'I feel so lonely .',
                  'I saw my lovely mom in a dream .',
                  'I am depressed .',
                  'you are laughing .',
                  'he was upset .',
                  'he told me that he was lonely .',
                  'he is upset .',
                  'it was a joyful reunion of all the family .',
                  'They laugh together .',
                  'he committed suicide yesterday .',
                  'he was sad and drunk at the end of the party .',
                  'she was grieving .',
                  'man runs software .',
                  'he was in a confused state of mind .',
                  'she was joyful .',
                  'she is upset .',
                  'I feel ashamed of my failure .',
                  'he was crying in the shower alone .',
                  'this decision is likely to upset a lot of people .',
                  'he was too upset to be rational .',
                  'I cry because I failed my exam .',
                  'the great woman makes dinner .',
                  'she was hurt .',
                  'she started to cry .',
                  'the depressed person was very sad and lonely .',
                  'he loves the figure from the board .',
                  'he loves to play .',
                  'you are doing good .',
                  'person cooks great meal .',
                  'we would appreciate it .',
                  'I mourn him .',
                  'this suit is perfect for me .',
                  'we must allow for his success .',
                  'I am happy to see you .',
                  'I smooched with him on the dance floor .',
                  'she was feeling great because of her dream .',
                  'she committed suicide  .',
                  'she was cooking a great fish as a treat .',
                  'sorrow and rage pierced him to the core .',
                  'they are depressed .',
                  'she was desperately lonely .',
                  'he is a successful man .',
                  'woman kills herself .',
                  'she looked about to cry .',
                  'I fell into misery .',
                  'it is a sad and depressing world .',
                  'man killed himself .',
                  "she's crying alone in her room .",
                  'I hated to make her sad .',
                  'woman was kind .',
                  'man kills himself .',
                  'she was upset for crying .',
                  'we must try to forget this sad affair .',
                  'she is very depressed .',
                  'she was ashamed of her background .',
                  'he is depressed .',
                  'I have a right to be upset .',
                  'she felt like a failure at her school .',
                  'she was happy .',
                  'life is a lovely dream .',
                  'he expressed sadness .',
                  'happy woman is smiling .',
                  'I am feeling amazing .',
                  'I am happy to talk .',
                  'I feel suicidal because I am depressed .',
                  'all the evidence points to suicide .',
                  'I read the love letter .',
                  'woman bakes tasty meal .',
                  'she committed suiced today .',
                  'she is the perfect woman .',
                  'sorrowful woman shed tears .',
                  'he wants to dance .',
                  'I was glad to have a success .',
                  'I was happy to see him .',
                  'I am depressed .',
                  'the man was doing amazing .',
                  'she was going to kill herself .',
                  'I am sad about something .',
                  'she was joyful of her good result .',
                  'I am very happy .',
                  'he felt like a failure .',
                  'she is upset .',
                  'kind woman helps student .',
                  'I share a happy dream with my mom .',
                  'I am sad about something .',
                  'I hate to see you upset .',
                  'she began to cry .',
                  'he would be upset .',
                  'I am a failure .',
                  'I should add that we are very lucky .',
                  'her anxiety mounted month by month .',
                  'I want to dance with her .',
                  'I am sorry they upset you .',
                  'she was having a sad nightmare .',
                  'I am cooking a good meal .',
                  'they are depressed .',
                  'I treat my family good .',
                  'she commited suicide in her room .',
                  'I am a good person .',
                  'I feel lucky today .',
                  'the road will lead you back to success .',
                  'I had a great surprise .',
                  'all of a sudden we heard a depressing cry .',
                  'we were shocked by his attempted suicide .',
                  'he has depression .'
                  'they take a class .']

    # Test data sentences
    test_data = ['someone laughs .',
                 'he has exhibited symptoms of anxiety .',
                 'you are sad about it .',
                 'woman began to cry .',
                 'happy man is laughing with happiness .',
                 'he felt remorse for his crimes .',
                 'her suicide attempt was really a cry for help .',
                 'she would like to have so much happiness .',
                 'depression can be traced to holding in anger .',
                 'she loves to play games .',
                 'she was angery and sad .',
                 'I had a joyful feeling .',
                 'we had to smile for the photo .',
                 'some babies cry during the night .',
                 'he is a very good man .',
                 'she suffered agonies of remorse .',
                 'she read the love letter .',
                 'I am crying .',
                 'she started to cry again .',
                 'you need to be a good man .',
                 'my life is like a happy dream .',
                 'I am upset .',
                 'I had depression .',
                 'I am feeling great .',
                 'she appreciates it .',
                 'they have partnered up for the lovely dance .',
                 'I had depression .',
                 'she loves him very much .',
                 'she is very depressed .',
                 'she had repeatedly threatened to commit suicide .',
                 'we all like to laugh .',
                 'I help my friends to move .',
                 'she writes great love letters .',
                 'I will try not to disappoint you next time .',
                 'she has depression .',
                 'she is laughing with happiness .',
                 'life is a depressing dream .',
                 "she's unhappy because he upset her .",
                 'you should consider smiling .',
                 'I am glad about his new job .',
                 'sad man cries in his room .',
                 'she has depression .',
                 'I am very happy for my great sister .',
                 'she put a smile on her face .',
                 'person is kind and helps people .',
                 'she is crying .',
                 'we all felt sad about his depression .',
                 'I was very happy .',
                 'he felt very lonely .',
                 'you are happy .',
                 'she had attempted suicide .',
                 'I am unhappy in my life .',
                 'he smiles when he sees himself .',
                 'I am crying because I am sad .',
                 'he cries over his sad life .',
                 'she cries herself to sleep .',
                 'he is crying .',
                 'the boys wanted to play .',
                 'he starves for friendship .',
                 'person was very kind .']

    new_test_data = ['I love Lambeq .']

    # Diagram to be tokenized
    diagram = Diagram(dom=Ty(), cod=Ty('s'),
    boxes=[Word('I', Ty('n')),
           Word('see', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
           Cup(Ty('n'), Ty(Ob('n', z=1))),
           Word('the', Ty('n', Ob('n', z=-1))),
           Cup(Ty(Ob('n', z=-1)), Ty('n')), Word('photo', Ty('n')),
           Cup(Ty(Ob('n', z=-1)), Ty('n'))],
    offsets=[0, 1, 0, 2, 1, 2, 1])

    # Expected diagram
    expected_train_diagram = Diagram(dom=Ty(), cod=Ty('s'),
            boxes=[Word('I', Ty('n')),
                   Word('see', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
                   Cup(Ty('n'), Ty(Ob('n', z=1))),
                   Word('the', Ty('n', Ob('n', z=-1))),
                   Cup(Ty(Ob('n', z=-1)), Ty('n')), Word('UNK', Ty('n')),
                   Cup(Ty(Ob('n', z=-1)), Ty('n'))],
            offsets=[0, 1, 0, 2, 1, 2, 1])

    # Initializing handler
    handler = HandleUnknownWords(train_data, test_data)

    # Handling the train diagram (min_freq of 2)
    tokenized_train_diagram = handler.handle_train(2, diagram)

    # Diagram to be tokenized
    diagram = Diagram(dom=Ty(), cod=Ty('s'),
    boxes=[Word('I', Ty('n')),
           Word('see', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
           Cup(Ty('n'), Ty(Ob('n', z=1))),
           Word('the', Ty('n', Ob('n', z=-1))),
           Cup(Ty(Ob('n', z=-1)), Ty('n')), Word('photo', Ty('n')),
           Cup(Ty(Ob('n', z=-1)), Ty('n'))],
    offsets=[0, 1, 0, 2, 1, 2, 1])

    # Expected diagram
    expected_test_diagram = Diagram(dom=Ty(), cod=Ty('s'),
            boxes=[Word('I', Ty('n')),
                   Word('see', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
                   Cup(Ty('n'), Ty(Ob('n', z=1))),
                   Word('the', Ty('n', Ob('n', z=-1))),
                   Cup(Ty(Ob('n', z=-1)), Ty('n')), Word('UNK', Ty('n')),
                   Cup(Ty(Ob('n', z=-1)), Ty('n'))],
            offsets=[0, 1, 0, 2, 1, 2, 1])

    # Handling the test diagram
    tokenized_test_diagram = handler.handle_test(diagram)

    handler.replace_test_data(new_test_data)

    # Asserting for checking the handler
    assert tokenized_train_diagram == expected_train_diagram
    assert tokenized_test_diagram == expected_test_diagram
    assert dataset_to_words(new_test_data) == handler.test_data    
