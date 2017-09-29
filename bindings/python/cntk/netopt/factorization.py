import cntk
from cntk.ops.functions import BlockFunction
from cntk.variables import Parameter
from cntk.ops import times
from cntk.internal import _as_tuple
from cntk.layers.blocks import _initializer_for, _INFERRED, identity, UntestedBranchError  # helpers
from cntk.default_options import is_default_override, get_default_override, default_override_or

def svd_subprojection(matrix, k):
    '''
    Calculate svd of the matrix and produce a subprojection based on k

    Args:
        matrix : an input matrix        
        k (int): length of one dimension of the matrix after sub projection

    Returns:
        two matrices representing the original matrix after svd and subprojection.
    '''

    from numpy import dot, diag
    from numpy.linalg import svd

    # Decompose W into (U, s, V)
    U, s, V = svd(matrix, full_matrices=False)
       
    # Create two dense layers from this; one that takes U, one that takes
    # dot(s, V), but restrict them all to rank k, such that the result is a
    # k-rank subprojection
    W1 = U[:, :k]
    W2 = dot(diag(s[:k]), V[:k, :])

    return W1, W2

def factor_dense(model, factor_function, projection_function):
    '''
    Factor a dense model into subprojection based on the provided factor_function and the 
    projection_function. If no projection_funciton is specified, use svd decomposition. 

    Args:
        model                   : dense model.
        factor_function         : function to factor the dense model (e.g. svd)   
        projection_function     : function to reduce the size of the dense model
        
    Returns:
        a model that is factored and projected (reduced).
    '''
    fltr_dense = (lambda x: type(x) == cntk.Function and x.op_name == 'Dense')

    def dense_converter(model):
        W, b = model.W.value, model.b.value    
        ht, wdth = W.shape
        k = projection_function(W)        
        W1, W2 = factor_function(W) if factor_function else svd_subprojection(W, k)
      
        Ws = {'W1': W1, 'W2': W2}
        dfl = _dense_factored((k, wdth),
            init=Ws,
            activation=None,
            init_bias=b,
            name='_factored_model')(model.inputs[2])
        return dfl

    return cntk.misc.convert(model, fltr_dense, dense_converter)

def _dense_factored(shapes, #(shape1, shape2)
                  activation=default_override_or(identity),
                  init={'W1':None, 'W2':None},
                  input_rank=None,
                  map_rank=None,
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  name=''):
    '''
    Perform the new model creation using the factored inputs W1 and W2. 
    The returend function represents the new model.

    Args:
        shapes                  : dimensions of the input matrices.
        activation              : activation function used for the model.
        init                    : the two matrices corresponding to the factorization.
        input_rank              : rank of the input tensor.
        map_rank                : ???
        bias                    : bias for the model.
        init_bias               : initial bias value.
        name                    : name of the block function that creates the new model.
        
    Returns:
        a model that is factored and projected (reduced).
    '''

    # matthaip: Not sure how to handle input tensor of rank > 1
    # or selective flattening of ranks
    assert(input_rank is None and
           map_rank is None and
           all(isinstance(s,int) for s in list(shapes)))

    activation = get_default_override(cntk.layers.Dense, activation=activation)
    bias       = get_default_override(cntk.layers.Dense, bias=bias)
    init_bias  = get_default_override(cntk.layers.Dense, init_bias=init_bias)
    # how to use get_default_override for init parameeter?

    output_shape1 = _as_tuple(shapes[0])
    output_shape2 = _as_tuple(shapes[1])
    if input_rank is not None and map_rank is not None:
        raise ValueError("Dense: input_rank and map_rank cannot be specified at the same time.")


    # If input_rank not given then pass a single _INFERRED; map_rank if given will determine the input_rank.
    # The dimension inference may still create multiple axes.
    input_shape = _INFERRED

    # parameters bound to this Function
    #    init_weights = _initializer_for(init, Record(output_rank=output_rank))
    init_weights = init
    W1 = Parameter(input_shape + output_shape1, init=init_weights['W1'], name='W1')
    W2 = Parameter(output_shape1 + output_shape2, init=init_weights['W2'], name='W2')
    b = Parameter(              output_shape2, init=init_bias,    name='b') if bias else None

    # expression of this function
    @BlockFunction('_factored_model', name)
    def dense(x):
        r = times(x, W1)
        r = times(r, W2)
        if b:
            r = r + b
        if activation is not None:
            r = activation(r)
        return r
    return dense

# Reference for sklearn.tucker.hooi:
# https://hal.inria.fr/hal-01219316/document
