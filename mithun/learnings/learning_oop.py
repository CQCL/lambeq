class Person:
    def __init__(self, name, job,salary):
        self.name=name
        self.job=job
        self.salary =salary
    def give_raise(self, percent):
        self.salary=self.salary*percent
    def __repr__(self):
        return "[%s, %s]" % (self.name,self.salary)

class Manager(Person):
    def __init__(self,name,salary):
        Person.__init__(self,name,"mgr",salary)

    def give_raise(self,percent, bonus=2):
        return Person.give_raise(self,percent*bonus)

if __name__=="__main__":
    sue = Person(name="Sue",job="dev", salary=2000)
    marj = Manager(name="Marjorie", salary=2000)
    for obj in sue, marj:
        obj.give_raise(2)
        print(obj.__class__.__name__)
        for k, v in obj.__dict__.items():
            print(k,v)

