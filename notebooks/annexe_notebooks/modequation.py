"""
Author : Olivier Pannekoucke
Date : 2020/11/2
Licence : CeCILL-B
"""


import sympy
from sympy import symbols, Derivative, Function, Eq

t = symbols('t')

def remove_eval_derivative(expr):
    """
    Remove substituted derivative in an expression.
    :param expr:
    :return:
    """
    eval_terms = expr.atoms(sympy.Subs)
    eval_derivative = {}
    for eval_term in eval_terms:
        eval_derivative[eval_term] = eval_term.args[0].subs({
            key: value for key, value in zip(eval_term.args[1], eval_term.args[2])
        })
    return expr.subs(eval_derivative).doit()


def get_at_order(equation, u, coordinates, order):
    """ Return the initial scheme expanded at a given order """
    args = []
    for item in equation.args:
        tmp = item
        for xi in coordinates:
            dxi = symbols('d'+str(xi))
            tmp = tmp.series(dxi,0,order)            
            tmp = remove_eval_derivative(tmp)        
        args.append(tmp)
    
    equation = Eq(*args)
    
    trend = Derivative(u,t)
    return Eq(trend, sympy.solve(equation, trend)[0])    

def get_consistant_equation(equation, coordinates):
    """ Return the equation at the leading order """
    dxs = [symbols('d'+str(xi)) for xi in coordinates]
    args = []
    for item in equation.args:
        item = item.removeO()
        item = item.subs({dxi:sympy.Integer(0) for dxi in dxs})
        args.append(item)
    return Eq(*args)


def derivative_order(derivative):
    """ Rerturn the total order of derivative 
    
    .. Example:
    
    >>> derivative_order(Derivative(u,t))
    1
    >>> derivative_order(Derivative(u,t,2,x,2))
    4
    
    """    
    order = 0
    for item in derivative.args[1:]:
        order += item[1]
    return order

class ModifiedEquation(object):
    """ Computation of the modified equation of a numerical scheme 
    
    Desciption
    ----------

        This applies for univariate dynamics (evolution equation for a single function 
        of the time and space 1D/2D/3D.. possible)
    
    """ 
    
    
    def __init__(self, scheme, u, order=2):
        self.scheme = scheme
        self.u = u
        self.coordinates = u.args
        self.order = order
        self._modified_equation = None
    
    @property
    def modified_equation(self):
        if self._modified_equation is None:
            self._modified_equation = self._get_modified_equation()
        return self._modified_equation
    
    @property
    def taylored_equation(self):
        
        coordinates = self.u.args
        trend = Derivative(self.u,t)
        scheme = self.scheme
        order = self.order

        return get_at_order(scheme, self.u, coordinates, order)

    @property
    def consistant_equation(self):        
        return self.taylored_equation.subs({symbols('d'+str(xi)):sympy.Integer(0) for xi in self.coordinates})
    
    
    def _get_modified_equation(self):
        """ Computation of the modified equation """
        coordinates = self.u.args
        trend = Derivative(self.u,t)
        scheme = self.scheme
        order = self.order

        taylored_equation = self.taylored_equation

        consistant_equation = self.consistant_equation


        # 1. Détermination de la tendance à partir de la consistance
        subs_trend = {trend:sympy.solve(consistant_equation, trend)[0]}

        # 2. Détermination du dictionnaire de remplacement rangé par ordre de dérivation (permettant de remplacer par ordre décroissant)
        ho_derivatives = (derivative for derivative in taylored_equation.atoms(Derivative) if derivative_order(derivative)>1 )
        subs_derivatives = {}
        for derivative in ho_derivatives:

            order = derivative_order(derivative)

            subs = derivative
            for k in range(order):
                subs = subs.subs(subs_trend).doit()

            # Ajoute la dérivée substituée
            if order in subs_derivatives:
                subs_derivatives[order][derivative] = subs
            else:
                subs_derivatives[order] = {derivative:subs}    


        # 3. Expression de l'équation modifiée en remplacant par ordre décroissant
        modified_equation = taylored_equation
        for order in range(1+max([order for order in subs_derivatives]))[::-1]:        
            if order in subs_derivatives:
                modified_equation = modified_equation.subs(subs_derivatives[order])
        modified_equation

        # 4. Expression de la tendance
        modified_equation = Eq(trend, sympy.solve(modified_equation, trend)[0])

        return modified_equation
