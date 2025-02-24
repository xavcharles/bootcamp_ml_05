class Matrix:
    def __init__(self, data=None, shape=None):
        if (data != None and not isinstance(data, list)):
            raise TypeError("data should be a list of lists")
        if (shape != None and not isinstance(shape, tuple)):
            raise TypeError("shape should be a tuple")
        if (data == None and shape == None):
            raise ValueError("should create an instance of Matrix with at least a shape or data")
        if (data == None):
            self.data = [[0] * shape[1] for _ in range(shape[0])]
            self.shape = shape
        elif (shape == None):
            self.shape = (len(data), len(data[0]))
            self.data = data
        return
    
    def T(self):
        lst = [[self.data[i][j] for i in range(self.shape[0])] for j in range(self.shape[1])]
        return Matrix(lst)

    def __add__(self, op):
        # add : only matrices of same dimensions.
        if (self.shape == op.shape):
            new_data = [[self.data[i][j] + op.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
        return Matrix(new_data)
                
    def __radd__(self, op):
        # add & radd : only vectors of the same shape.
        if (self.shape == op.shape):
            new_data = [[op.data[i][j] + self.data[i][j] for j in range(op.shape[1])] for i in range(op.shape[0])]
        return Matrix(new_data)

    def __sub__(self, op):
        # sub : only matrices of same dimensions.
        if (self.shape == op.shape):
            new_data = [[self.data[i][j] - op.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
        return Matrix(new_data)
            
    def __rsub__(self, op):
        # sub & rsub: only vectors of the same shape.
        if (self.shape == op.shape):
            new_data = [[op.data[i][j] - self.data[i][j] for j in range(op.shape[1])] for i in range(op.shape[0])]
        return Matrix(new_data)

    def __truediv__(self, op):
        # truediv : only with scalars (to perform division of a Vector by a scalar).
        if (op == 0):
            print("ZeroDivisionError: division by zero")
            return
        if isinstance(op, (int, float)):
            new_data = [[self.data[i][j] / op for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Matrix(new_data)
        return NotImplementedError

    def __rtruediv__(self, op):
        # rtruediv : raises an NotImplementedError with the message "Division of a scalar by a Vector is not defined here."
        raise NotImplementedError("Division of a scalar by a Vector is not defined here.")

    def __mul__(self, op):
        print("test")
        if isinstance(op, (int, float)):
            new_data = [[self.data[i][j] * op for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Matrix(new_data)
        elif isinstance(op, Matrix):
            if (self.shape[1] == op.shape[0]):
                new_data = [[sum(self.data[j][i] * op.data[i][k] for i in range(self.shape[1])) for k in range(op.shape[1])] for j in range(self.shape[0])]
            elif (self.shape[0] == op.shape[1]):
                new_data = [[sum(self.data[i][k] * op.data[j][i] for i in range(self.shape[0])) for k in range(op.shape[0])] for j in range(self.shape[1])]
                return Matrix(new_data)
        elif isinstance(op, Vector):
            if (op.shape[1] == 1 and op.shape[0] == self.shape[1]):
                new_data = [[sum(self.data[j][i] * op.data[i][0] for i in range(self.shape[1]))] for j in range(self.shape[0])]
                return Vector(new_data)
            elif (op.shape[0] == 1 and op.shape[1] == self.shape[0]):
                new_data = [[sum(self.data[i][j] * op.data[0][i] for i in range(self.shape[0])) for j in range(self.shape[1])]]
                return Vector(new_data)
            else:
                return ValueError("vector dimensions not compatible with multiplication by this matrix")
        print("test")
        return NotImplemented

    def __rmul__(self, op):
        if isinstance(op, (int, float)):
            new_data = [[self.data[i][j] * op for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Matrix(new_data)
        elif isinstance(op, Matrix):
            if (op.shape[1] == self.shape[0]):
                new_data = [[sum(op.data[j][i] * self.data[i][k] for i in range(op.shape[1])) for k in range(self.shape[1])] for j in range(op.shape[0])]
                return Matrix(new_data)
        elif isinstance(op, Vector):
            if (op.shape[1] == 1 and op.shape[0] == self.shape[1]):
                new_data = [[sum(self.data[j][i] * op.data[i][0] for i in range(self.shape[1]))] for j in range(self.shape[0])]
                return Vector(new_data)
            elif (op.shape[0] == 1 and op.shape[1] == self.shape[0]):
                new_data = [[sum(self.data[i][j] * op.data[0][i] for i in range(self.shape[0])) for j in range(self.shape[1])]]
                return Vector(new_data)
            else:
                return ValueError("vector dimensions not compatible with multiplication by this matrix")
        # elif isinstance(op, Vector):
        #     if (op.shape[1] == 1 and op.shape[0] == self.shape[1]):
        #         new_data = [[sum(self.data[j][i] * op.data[i][0]) for i in range(self.shape[1])] for j in range(self.shape[0])]
        #         return Vector(new_data)
        #     elif (op.shape[0] == 1 and op.shape[1] == op.shape[0]):
        #         new_data = [[sum(self.data[i][j] * op.data[0][i] for i in range(self.shape[0])) for j in range(self.shape[1])]]
        #         return Vector(new_data)
        #     else:
        #         return ValueError("vector dimensions not compatible with multiplication by this matrix")
        return NotImplemented

    def __str__(self):
        return (f"{self.data}")

    def __repr__(self):
        # must be identical, i.e we expect that print(vector) and vector within python interpretor to behave the same, see correspo
        return (f"{type(self).__name__}({self.data})")
    
class Vector(Matrix):
    def __init__(self, data=None):
        super().__init__(data=data)
        # if (shape != None and not (shape[0] == 1 or shape[1] == 1)):
        #     return ValueError("A Vector can only be uni-dimensional")
        if (data != None):
            if len(data) > 1:
                if not (all(isinstance(sublist, list) and len(sublist) == 1 for sublist in data)):
                    raise TypeError("This is either not a list of lists or some of the lists do not have exactly one element")
                self.data = data
                self.shape = (len(self.data), 1)
            elif (len(data) == 1):
                if not (isinstance(data[0], list) and all(isinstance(data[0][i], int) or isinstance(data[0][i], float) for i in range(len(data[0])))):
                    raise TypeError("This is either not a list of a list or not all elements of the list of a list are floats or ints")
                self.data = data
                self.shape = (1, len(self.data[0]))
        else:
            return ValueError("gotta give me something to create the vector, mate")
