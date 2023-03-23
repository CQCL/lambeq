import discopy.rigid


class cat:
    def __init__(self):
        from discopy.cat import Ob, Box, Id

        A, B, C, D = Ob('A'), Ob('B'), Ob('C'), Ob('D')
        f= Box('f',A,B)
        g= Box('g',B,C)
        h= Box('h',C,D)

        print(type(f.cod))

        assert g<<f == f>>g
        assert h << (g<<f) == (h <<g) << f
        assert Id(B) << f == f == f << Id(A)
        assert f<<Id(A)==f==Id(B)<<f

        # arrow = h<<g<<f
        #test8:wq
        # print(repr(arrow))
        # # print(repr(arrow))
        # print(arrow[1:])
        # print(repr(sorted(arrow,key=lambda x: x[0],reverse=True)))
        # print((arrow[::-1]))

class mono:
        def __init__(self):
            from discopy.monoidal import Box, Id, Ty


            A, B, C, D  =Ty('A'),Ty('B'), Ty('C'), Ty('D')


            ######Combining objects
            # q = A @ B @ C
            # print(repr(q))
            # print(repr(q[::-1]))
            #
            # ####how to get a Ty() from a list/composition of Ty()
            # print(type(q[1]))
            # print(type(q[0:1]))


            ##Combining morphisms/boxes with Ty using @ and  then drawing it using >>
            #refer this for FAQ: https://docs.google.com/document/d/16Sfz3O7MHd7uQcMlW4FxzxlaUOZp0tgWu4FSwcCD6BA/edit?usp=sharing
            #
            # x = Box('x', A, A)
            # # print(x.dom)
            # # print(x.cod)
            #
            #
            # y = Box('y', A@A, B)
            # z=x@Id(A)

            # print(z.dom)
            # print(z.cod)
            # print(y.dom)
            # print(y.cod)

            # arrow = z>>y
            # print(repr(arrow))
            # print(type(arrow))
            # arrow.draw()



            #slicing diagrams
            # arrow=arrow[1:]
            #arrow.draw()
            # print(repr(arrow))
            # print(type(arrow))

            #reversing diagrams
            # arrow = arrow[::-1]
            # arrow.draw()
            # print(repr(arrow))
            #print(type(arrow))

            #creating diagrams from class Diagram
            # from discopy import Diagram
            # d1 = Diagram(dom=Ty('A', 'A'), cod=Ty('B'),
            #             boxes=[Box('x', Ty('A'), Ty('A')), Box('y', Ty('A', 'A'), Ty('B'))], offsets=[0, 0])
            # d2=Diagram(dom=Ty('A', 'A'), cod=Ty('B'), boxes=[Box('x', Ty('A'), Ty('A')), Box('y', Ty('A', 'A'), Ty('B'))], offsets=[1, 0])
            # d1.draw()
            # d2.draw()

            #sswaps
            # from discopy.monoidal import Swap
            # x=Swap(d1,d2)
            # print(repr(x))
            # print(type(x))
            # x.draw()

            # #cups and caps
            # from discopy.rigid import Id, Ty, Cup, Cap, Box
            # a=Ty('A')
            #
            # kyaapAr=Cap(a.r,a)
            # idkyaapAr = Id(a) @ kyaapAr
            # kyaapAr.draw()
            # idkyaapAr.draw()
            # kyupAr = Cup(A,A.r)
            # kyupArId =kyupAr @ Id(A)
            # self.printDomainCodomain(kyupArId)
            #
            #
            # #snake
            # snaayke=idkyaapAr>>kyupArId
            #
            # #snake1 = Id(A) @ Cap(A.r, A) >> Cup(A, A.r) @ Id(A)
            # snaayke.draw(figsize=(8,8))


            #pregroup grammar- words
            from discopy import grammar,Box
            from discopy.grammar import Word
            from discopy.rigid import Ty, Cup, Cap, Id

            #john likes mary
            s , n = Ty('s'), Ty('n')
            j=Word('John',n)
            l=Word('likes',n.r@s@n.l)
            m=Word('Mary',n)
            Words=[j,l,m]


            cups_chain=Cup(n,n.r)@Id(s) @Cup(n.l,n)
            d=Id().tensor(*Words)>> cups_chain
            d.draw()
            grammar.draw(d)
            # cups_chain.draw().




        def printDomainCodomain(self,a):
            assert type(a) in [discopy.rigid.Ty, discopy.Diagram]
            print(f"Domain of given object {repr(a)} of type {type(a)} is: {a.dom} and its codomain is {a.cod}")
        def printObjType(self,a):
            print(f"Given object is of type {type(a)} ")

        def printAdjoint(self,a):
            assert type(a) in [discopy.rigid.Ty, discopy.Diagram]
            print(f"Given object {a} whose left adjoint is: {a.l} and its right adjoint is {a.r}")











if __name__=="__main__":
    mono()
