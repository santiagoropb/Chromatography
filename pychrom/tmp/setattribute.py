import weakref

class Model(object):


    def __init__(self, value):
        self.value = value
        self.b = Blah('my blah2')
        print(self.b.model().b.value)


    def __setattr__(self, name, value):
        if isinstance(value, Blah):
            value.model = weakref.ref(self)
        super(Model, self).__setattr__(name, value)


class Blah(object):
    def __init__(self, value):
        self.value = value
        self.model = None


if __name__ == '__main__':
    m = Model('My Model')
    m.a = Blah('My Blah')

    print(m.a.value)
    print(m.a.model().value)
