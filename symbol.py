from typing import Any
from typing import TYPE_CHECKING, overload

from typing import ClassVar, TypeVar, Any, Self

Tbasic = TypeVar("Tbasic", bound='Basic')

def cacheit(maxsize):
    """caching decorator.

        important: the result of cached function must be *immutable*


        Examples
        ========

        >>> from sympy import cacheit
        >>> @cacheit
        ... def f(a, b):
        ...    return a+b

        >>> @cacheit
        ... def f(a, b): # noqa: F811
        ...    return [a, b] # <-- WRONG, returns mutable object

        to force cacheit to check returned results mutability and consistency,
        set environment variable SYMPY_USE_CACHE to 'debug'
    """
    def func_wrapper(func):
        cfunc = lru_cache(maxsize, typed=True)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                retval = cfunc(*args, **kwargs)
            except TypeError as e:
                if not e.args or not e.args[0].startswith('unhashable type:'):
                    raise
                retval = func(*args, **kwargs)
            return retval

        wrapper.cache_info = cfunc.cache_info
        wrapper.cache_clear = cfunc.cache_clear

        CACHE.append(wrapper)
        return wrapper

    return func_wrapper

class KindMeta(type):
    """
    Metaclass for ``Kind``.

    Assigns empty ``dict`` as class attribute ``_inst`` for every class,
    in order to endow singleton-like behavior.
    """
    def __new__(cls, clsname, bases, dct):
        dct['_inst'] = {}
        return super().__new__(cls, clsname, bases, dct)


class Kind(object, metaclass=KindMeta):
    """
    Base class for kinds.

    Kind of the object represents the mathematical classification that
    the entity falls into. It is expected that functions and classes
    recognize and filter the argument by its kind.

    Kind of every object must be carefully selected so that it shows the
    intention of design. Expressions may have different kind according
    to the kind of its arguments. For example, arguments of ``Add``
    must have common kind since addition is group operator, and the
    resulting ``Add()`` has the same kind.

    For the performance, each kind is as broad as possible and is not
    based on set theory. For example, ``NumberKind`` includes not only
    complex number but expression containing ``S.Infinity`` or ``S.NaN``
    which are not strictly number.

    Kind may have arguments as parameter. For example, ``MatrixKind()``
    may be constructed with one element which represents the kind of its
    elements.

    ``Kind`` behaves in singleton-like fashion. Same signature will
    return the same object.

    """
    def __new__(cls, *args):
        if args in cls._inst:
            inst = cls._inst[args]
        else:
            inst = super().__new__(cls)
            cls._inst[args] = inst
        return inst


class _UndefinedKind(Kind):
    """
    Default kind for all SymPy object. If the kind is not defined for
    the object, or if the object cannot infer the kind from its
    arguments, this will be returned.

    Examples
    ========

    >>> from sympy import Expr
    >>> Expr().kind
    UndefinedKind
    """
    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return "UndefinedKind"

UndefinedKind = _UndefinedKind()

class FactKB(dict):
    """
    A simple propositional knowledge base relying on compiled inference rules.
    """
    def __str__(self):
        return '{\n%s}' % ',\n'.join(
            ["\t%s: %s" % i for i in sorted(self.items())])

    def __init__(self, rules):
        self.rules = rules

    def _tell(self, k, v):
        """Add fact k=v to the knowledge base.

        Returns True if the KB has actually been updated, False otherwise.
        """
        if k in self and self[k] is not None:
            if self[k] == v:
                return False
            else:
                raise InconsistentAssumptions(self, k, v)
        else:
            self[k] = v
            return True

    # *********************************************
    # * This is the workhorse, so keep it *fast*. *
    # *********************************************
    def deduce_all_facts(self, facts):
        """
        Update the KB with all the implications of a list of facts.

        Facts can be specified as a dictionary or as a list of (key, value)
        pairs.
        """
        # keep frequently used attributes locally, so we'll avoid extra
        # attribute access overhead
        full_implications = self.rules.full_implications
        beta_triggers = self.rules.beta_triggers
        beta_rules = self.rules.beta_rules

        if isinstance(facts, dict):
            facts = facts.items()

        while facts:
            beta_maytrigger = set()

            # --- alpha chains ---
            for k, v in facts:
                if not self._tell(k, v) or v is None:
                    continue

                # lookup routing tables
                for key, value in full_implications[k, v]:
                    self._tell(key, value)

                beta_maytrigger.update(beta_triggers[k, v])

            # --- beta chains ---
            facts = []
            for bidx in beta_maytrigger:
                bcond, bimpl = beta_rules[bidx]
                if all(self.get(k) is v for k, v in bcond):
                    facts.append(bimpl)

class StdFactKB(FactKB):
    """A FactKB specialized for the built-in rules

    This is the only kind of FactKB that Basic objects should use.
    """
    def __init__(self, facts=None):
        super().__init__(_assume_rules)
        # save a copy of the facts dict
        if not facts:
            self._generator = {}
        elif not isinstance(facts, FactKB):
            self._generator = facts.copy()
        else:
            self._generator = facts.generator
        if facts:
            self.deduce_all_facts(facts)

    def copy(self):
        return self.__class__(self)

    @property
    def generator(self):
        return self._generator.copy()


def as_property(fact):
    """Convert a fact name to the name of the corresponding property"""
    return 'is_%s' % fact


def make_property(fact):
    """Create the automagic property corresponding to a fact."""

    def getit(self):
        try:
            return self._assumptions[fact]
        except KeyError:
            if self._assumptions is self.default_assumptions:
                self._assumptions = self.default_assumptions.copy()
            return _ask(fact, self)

    getit.func_name = as_property(fact)
    return property(getit)


class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way. Natively Python uses ``repr()``
    even if ``str()`` was explicitly requested. Mix in this trait into
    a class to get proper default printing.

    This also adds support for LaTeX printing in jupyter notebooks.
    """

    # Since this class is used as a mixin we set empty slots. That means that
    # instances of any subclasses that use slots will not need to have a
    # __dict__.
    __slots__ = ()

    # Note, we always use the default ordering (lex) in __str__ and __repr__,
    # regardless of the global setting. See issue 5487.
    def __str__(self):
        from sympy.printing.str import sstr
        return sstr(self, order=None)

    __repr__ = __str__

    def _repr_disabled(self):
        """
        No-op repr function used to disable jupyter display hooks.

        When :func:`sympy.init_printing` is used to disable certain display
        formats, this function is copied into the appropriate ``_repr_*_``
        attributes.

        While we could just set the attributes to `None``, doing it this way
        allows derived classes to call `super()`.
        """
        return None

    # We don't implement _repr_png_ here because it would add a large amount of
    # data to any notebook containing SymPy expressions, without adding
    # anything useful to the notebook. It can still enabled manually, e.g.,
    # for the qtconsole, with init_printing().
    _repr_png_ = _repr_disabled

    _repr_svg_ = _repr_disabled

    def _repr_latex_(self):
        """
        IPython/Jupyter LaTeX printing

        To change the behavior of this (e.g., pass in some settings to LaTeX),
        use init_printing(). init_printing() will also enable LaTeX printing
        for built in numeric types like ints and container types that contain
        SymPy objects, like lists and dictionaries of expressions.
        """
        from sympy.printing.latex import latex
        s = latex(self, mode='plain')
        return "$\\displaystyle %s$" % s




class Basic(Printable):
    """
    Base class for all SymPy objects.

    Notes and conventions
    =====================

    1) Always use ``.args``, when accessing parameters of some instance:

    >>> from sympy import cot
    >>> from sympy.abc import x, y

    >>> cot(x).args
    (x,)

    >>> cot(x).args[0]
    x

    >>> (x*y).args
    (x, y)

    >>> (x*y).args[1]
    y


    2) Never use internal methods or variables (the ones prefixed with ``_``):

    >>> cot(x)._args    # do not use this, use cot(x).args instead
    (x,)


    3)  By "SymPy object" we mean something that can be returned by
        ``sympify``.  But not all objects one encounters using SymPy are
        subclasses of Basic.  For example, mutable objects are not:

        >>> from sympy import Basic, Matrix, sympify
        >>> A = Matrix([[1, 2], [3, 4]]).as_mutable()
        >>> isinstance(A, Basic)
        False

        >>> B = sympify(A)
        >>> isinstance(B, Basic)
        True
    """
    __slots__ = ('_mhash',              # hash value
                 '_args',               # arguments
                 '_assumptions'
                )

    _args: tuple[...]
    _mhash: int | None

    @property
    def __sympy__(self):
        return True

    def __init_subclass__(cls):
        # Initialize the default_assumptions FactKB and also any assumptions
        # property methods. This method will only be called for subclasses of
        # Basic but not for Basic itself so we call
        # _prepare_class_assumptions(Basic) below the class definition.
        super().__init_subclass__()
        _prepare_class_assumptions(cls)

    # To be overridden with True in the appropriate subclasses
    is_number = False
    is_Atom = False
    is_Symbol = False
    is_symbol = False
    is_Indexed = False
    is_Dummy = False
    is_Wild = False
    is_Function = False
    is_Add = False
    is_Mul = False
    is_Pow = False
    is_Number = False
    is_Float = False
    is_Rational = False
    is_Integer = False
    is_NumberSymbol = False
    is_Order = False
    is_Derivative = False
    is_Piecewise = False
    is_Poly = False
    is_AlgebraicNumber = False
    is_Relational = False
    is_Equality = False
    is_Boolean = False
    is_Not = False
    is_Matrix = False
    is_Vector = False
    is_Point = False
    is_MatAdd = False
    is_MatMul = False

    default_assumptions: ClassVar[StdFactKB]

    is_composite: bool | None
    is_noninteger: bool | None
    is_extended_positive: bool | None
    is_negative: bool | None
    is_complex: bool | None
    is_extended_nonpositive: bool | None
    is_integer: bool | None
    is_positive: bool | None
    is_rational: bool | None
    is_extended_nonnegative: bool | None
    is_infinite: bool | None
    is_antihermitian: bool | None
    is_extended_negative: bool | None
    is_extended_real: bool | None
    is_finite: bool | None
    is_polar: bool | None
    is_imaginary: bool | None
    is_transcendental: bool | None
    is_extended_nonzero: bool | None
    is_nonzero: bool | None
    is_odd: bool | None
    is_algebraic: bool | None
    is_prime: bool | None
    is_commutative: bool | None
    is_nonnegative: bool | None
    is_nonpositive: bool | None
    is_hermitian: bool | None
    is_irrational: bool | None
    is_real: bool | None
    is_zero: bool | None
    is_even: bool | None

    kind: Kind = UndefinedKind

    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._assumptions = cls.default_assumptions
        obj._mhash = None  # will be set by __hash__ method.

        obj._args = args  # all items in args must be Basic objects
        return obj

    def copy(self):
        return self.func(*self.args)

    def __getnewargs__(self):
        return self.args

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __reduce_ex__(self, protocol):
        if protocol < 2:
            msg = "Only pickle protocol 2 or higher is supported by SymPy"
            raise NotImplementedError(msg)
        return super().__reduce_ex__(protocol)

    def __hash__(self) -> int:
        # hash cannot be cached using cache_it because infinite recurrence
        # occurs as hash is needed for setting cache dictionary keys
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h

    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.

        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args

    @property
    def assumptions0(self):
        """
        Return object `type` assumptions.

        For example:

          Symbol('x', real=True)
          Symbol('x', integer=True)

        are different objects. In other words, besides Python type (Symbol in
        this case), the initial assumptions are also forming their typeinfo.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.abc import x
        >>> x.assumptions0
        {'commutative': True}
        >>> x = Symbol("x", positive=True)
        >>> x.assumptions0
        {'commutative': True, 'complex': True, 'extended_negative': False,
         'extended_nonnegative': True, 'extended_nonpositive': False,
         'extended_nonzero': True, 'extended_positive': True, 'extended_real':
         True, 'finite': True, 'hermitian': True, 'imaginary': False,
         'infinite': False, 'negative': False, 'nonnegative': True,
         'nonpositive': False, 'nonzero': True, 'positive': True, 'real':
         True, 'zero': False}
        """
        return {}

    def compare(self, other):
        """
        Return -1, 0, 1 if the object is less than, equal,
        or greater than other in a canonical sense.
        Non-Basic are always greater than Basic.
        If both names of the classes being compared appear
        in the `ordering_of_classes` then the ordering will
        depend on the appearance of the names there.
        If either does not appear in that list, then the
        comparison is based on the class name.
        If the names are the same then a comparison is made
        on the length of the hashable content.
        Items of the equal-lengthed contents are then
        successively compared using the same rules. If there
        is never a difference then 0 is returned.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x.compare(y)
        -1
        >>> x.compare(x)
        0
        >>> y.compare(x)
        1

        """
        # all redefinitions of __cmp__ method should start with the
        # following lines:
        if self is other:
            return 0
        n1 = self.__class__
        n2 = other.__class__
        c = _cmp_name(n1, n2)
        if c:
            return c
        #
        st = self._hashable_content()
        ot = other._hashable_content()
        len_st = len(st)
        len_ot = len(ot)
        c = (len_st > len_ot) - (len_st < len_ot)
        if c:
            return c
        for l, r in zip(st, ot):
            if isinstance(l, Basic):
                c = l.compare(r)
            elif isinstance(l, frozenset):
                l = Basic(*l) if isinstance(l, frozenset) else l
                r = Basic(*r) if isinstance(r, frozenset) else r
                c = l.compare(r)
            else:
                c = (l > r) - (l < r)
            if c:
                return c
        return 0

    @classmethod
    def fromiter(cls, args, **assumptions):
        """
        Create a new object from an iterable.

        This is a convenience function that allows one to create objects from
        any iterable, without having to convert to a list or tuple first.

        Examples
        ========

        >>> from sympy import Tuple
        >>> Tuple.fromiter(i for i in range(5))
        (0, 1, 2, 3, 4)

        """
        return cls(*tuple(args), **assumptions)

    @classmethod
    def class_key(cls) -> tuple[int, int, str]:
        """Nice order of classes."""
        return 5, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        """
        Return a sort key.

        Examples
        ========

        >>> from sympy import S, I

        >>> sorted([S(1)/2, I, -I], key=lambda x: x.sort_key())
        [1/2, -I, I]

        >>> S("[x, 1/x, 1/x**2, x**2, x**(1/2), x**(1/4), x**(3/2)]")
        [x, 1/x, x**(-2), x**2, sqrt(x), x**(1/4), x**(3/2)]
        >>> sorted(_, key=lambda x: x.sort_key())
        [x**(-2), 1/x, x**(1/4), sqrt(x), x, x**(3/2), x**2]

        """

        # XXX: remove this when issue 5169 is fixed
        def inner_key(arg):
            if isinstance(arg, Basic):
                return arg.sort_key(order)
            else:
                return arg

        args = self._sorted_args
        args = len(args), tuple([inner_key(arg) for arg in args])
        return self.class_key(), args, S.One.sort_key(), S.One

    def _do_eq_sympify(self, other):
        """Returns a boolean indicating whether a == b when either a
        or b is not a Basic. This is only done for types that were either
        added to `converter` by a 3rd party or when the object has `_sympy_`
        defined. This essentially reuses the code in `_sympify` that is
        specific for this use case. Non-user defined types that are meant
        to work with SymPy should be handled directly in the __eq__ methods
        of the `Basic` classes it could equate to and not be converted. Note
        that after conversion, `==`  is used again since it is not
        necessarily clear whether `self` or `other`'s __eq__ method needs
        to be used."""
        for superclass in type(other).__mro__:
            conv = _external_converter.get(superclass)
            if conv is not None:
                return self == conv(other)
        if hasattr(other, '_sympy_'):
            return self == other._sympy_()
        return NotImplemented

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        their symbolic trees.

        This is the same as a.compare(b) == 0 but faster.

        Notes
        =====

        If a class that overrides __eq__() needs to retain the
        implementation of __hash__() from a parent class, the
        interpreter must be told this explicitly by setting
        __hash__ : Callable[[object], int] = <ParentClass>.__hash__.
        Otherwise the inheritance of __hash__() will be blocked,
        just as if __hash__ had been explicitly set to None.

        References
        ==========

        from https://docs.python.org/dev/reference/datamodel.html#object.__hash__
        """
        if self is other:
            return True

        if not isinstance(other, Basic):
            return self._do_eq_sympify(other)

        # check for pure number expr
        if  not (self.is_Number and other.is_Number) and (
                type(self) != type(other)):
            return False
        a, b = self._hashable_content(), other._hashable_content()
        if a != b:
            return False
        # check number *in* an expression
        for a, b in zip(a, b):
            if not isinstance(a, Basic):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True

    def __ne__(self, other):
        """``a != b``  -> Compare two symbolic trees and see whether they are different

        this is the same as:

        ``a.compare(b) != 0``

        but faster
        """
        return not self == other

    def dummy_eq(self, other, symbol=None):
        """
        Compare two expressions and handle dummy symbols.

        Examples
        ========

        >>> from sympy import Dummy
        >>> from sympy.abc import x, y

        >>> u = Dummy('u')

        >>> (u**2 + 1).dummy_eq(x**2 + 1)
        True
        >>> (u**2 + 1) == (x**2 + 1)
        False

        >>> (u**2 + y).dummy_eq(x**2 + y, x)
        True
        >>> (u**2 + y).dummy_eq(x**2 + y, y)
        False

        """
        s = self.as_dummy()
        o = _sympify(other)
        o = o.as_dummy()

        dummy_symbols = [i for i in s.free_symbols if i.is_Dummy]

        if len(dummy_symbols) == 1:
            dummy = dummy_symbols.pop()
        else:
            return s == o

        if symbol is None:
            symbols = o.free_symbols

            if len(symbols) == 1:
                symbol = symbols.pop()
            else:
                return s == o

        tmp = dummy.__class__()

        return s.xreplace({dummy: tmp}) == o.xreplace({symbol: tmp})

    @overload
    def atoms(self) -> set[Self]: ...
    @overload
    def atoms(self, *types: Tbasic | type[Tbasic]) -> set[Tbasic]: ...

    def atoms(self, *types: Tbasic | type[Tbasic]) -> set[Self] | set[Tbasic]:
        """Returns the atoms that form the current object.

        By default, only objects that are truly atomic and cannot
        be divided into smaller pieces are returned: symbols, numbers,
        and number symbols like I and pi. It is possible to request
        atoms of any type, however, as demonstrated below.

        Examples
        ========

        >>> from sympy import I, pi, sin
        >>> from sympy.abc import x, y
        >>> (1 + x + 2*sin(y + I*pi)).atoms()
        {1, 2, I, pi, x, y}

        If one or more types are given, the results will contain only
        those types of atoms.

        >>> from sympy import Number, NumberSymbol, Symbol
        >>> (1 + x + 2*sin(y + I*pi)).atoms(Symbol)
        {x, y}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number)
        {1, 2}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol)
        {1, 2, pi}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol, I)
        {1, 2, I, pi}

        Note that I (imaginary unit) and zoo (complex infinity) are special
        types of number symbols and are not part of the NumberSymbol class.

        The type can be given implicitly, too:

        >>> (1 + x + 2*sin(y + I*pi)).atoms(x) # x is a Symbol
        {x, y}

        Be careful to check your assumptions when using the implicit option
        since ``S(1).is_Integer = True`` but ``type(S(1))`` is ``One``, a special type
        of SymPy atom, while ``type(S(2))`` is type ``Integer`` and will find all
        integers in an expression:

        >>> from sympy import S
        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(1))
        {1}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(2))
        {1, 2}

        Finally, arguments to atoms() can select more than atomic atoms: any
        SymPy type (loaded in core/__init__.py) can be listed as an argument
        and those types of "atoms" as found in scanning the arguments of the
        expression recursively:

        >>> from sympy import Function, Mul
        >>> from sympy.core.function import AppliedUndef
        >>> f = Function('f')
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(Function)
        {f(x), sin(y + I*pi)}
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(AppliedUndef)
        {f(x)}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Mul)
        {I*pi, 2*sin(y + I*pi)}

        """
        nodes = _preorder_traversal(self)
        if types:
            types2 = tuple([t if isinstance(t, type) else type(t) for t in types])
            return {node for node in nodes if isinstance(node, types2)}
        else:
            return {node for node in nodes if not node.args}

    @property
    def free_symbols(self) -> set[Self]:
        """Return from the atoms of self those which are free symbols.

        Not all free symbols are ``Symbol`` (see examples)

        For most expressions, all symbols are free symbols. For some classes
        this is not true. e.g. Integrals use Symbols for the dummy variables
        which are bound variables, so Integral has a method to return all
        symbols except those. Derivative keeps track of symbols with respect
        to which it will perform a derivative; those are
        bound variables, too, so it has its own free_symbols method.

        Any other method that uses bound variables should implement a
        free_symbols method.

        Examples
        ========

        >>> from sympy import Derivative, Integral, IndexedBase
        >>> from sympy.abc import x, y, n
        >>> (x + 1).free_symbols
        {x}
        >>> Integral(x, y).free_symbols
        {x, y}

        Not all free symbols are actually symbols:

        >>> IndexedBase('F')[0].free_symbols
        {F, F[0]}

        The symbols of differentiation are not included unless they
        appear in the expression being differentiated.

        >>> Derivative(x + y, y).free_symbols
        {x, y}
        >>> Derivative(x, y).free_symbols
        {x}
        >>> Derivative(x, (y, n)).free_symbols
        {n, x}

        If you want to know if a symbol is in the variables of the
        Derivative you can do so as follows:

        >>> Derivative(x, y).has_free(y)
        True
        """
        empty: set[Basic] = set()
        return empty.union(*(a.free_symbols for a in self.args))

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        return set()

    def as_dummy(self) -> "Self":
        """Return the expression with any objects having structurally
        bound symbols replaced with unique, canonical symbols within
        the object in which they appear and having only the default
        assumption for commutativity being True. When applied to a
        symbol a new symbol having only the same commutativity will be
        returned.

        Examples
        ========

        >>> from sympy import Integral, Symbol
        >>> from sympy.abc import x
        >>> r = Symbol('r', real=True)
        >>> Integral(r, (r, x)).as_dummy()
        Integral(_0, (_0, x))
        >>> _.variables[0].is_real is None
        True
        >>> r.as_dummy()
        _r

        Notes
        =====

        Any object that has structurally bound variables should have
        a property, ``bound_symbols`` that returns those symbols
        appearing in the object.
        """
        from .symbol import Dummy, Symbol
        def can(x):
            # mask free that shadow bound
            free = x.free_symbols
            bound = set(x.bound_symbols)
            d = {i: Dummy() for i in bound & free}
            x = x.subs(d)
            # replace bound with canonical names
            x = x.xreplace(x.canonical_variables)
            # return after undoing masking
            return x.xreplace({v: k for k, v in d.items()})
        if not self.has(Symbol):
            return self
        return self.replace(
            lambda x: hasattr(x, 'bound_symbols'),
            can,
            simultaneous=False) # type:ignore

    @property
    def canonical_variables(self) -> dict[Self, Symbol]:
        """Return a dictionary mapping any variable defined in
        ``self.bound_symbols`` to Symbols that do not clash
        with any free symbols in the expression.

        Examples
        ========

        >>> from sympy import Lambda
        >>> from sympy.abc import x
        >>> Lambda(x, 2*x).canonical_variables
        {x: _0}
        """
        bound: list[Basic] | None = getattr(self, 'bound_symbols', None)
        if bound is None:
            return {}
        dums = numbered_symbols('_')
        reps = {}
        # watch out for free symbol that are not in bound symbols;
        # those that are in bound symbols are about to get changed

        # XXX: free_symbols only returns particular kinds of expressions that
        # generally have a .name attribute. There is not a proper class/type
        # that represents this.
        names = {i.name for i in self.free_symbols - set(bound)} # type: ignore
        for b in bound:
            d = next(dums)
            if b.is_Symbol:
                while d.name in names:
                    d = next(dums)
            reps[b] = d
        return reps

    def rcall(self, *args):
        """Apply on the argument recursively through the expression tree.

        This method is used to simulate a common abuse of notation for
        operators. For instance, in SymPy the following will not work:

        ``(x+Lambda(y, 2*y))(z) == x+2*z``,

        however, you can use:

        >>> from sympy import Lambda
        >>> from sympy.abc import x, y, z
        >>> (x + Lambda(y, 2*y)).rcall(z)
        x + 2*z
        """
        if callable(self):
            return self(*args)
        elif self.args:
            newargs = [sub.rcall(*args) for sub in self.args]
            return self.func(*newargs)
        else:
            return self

    def is_hypergeometric(self, k):
        from sympy.simplify.simplify import hypersimp
        from sympy.functions.elementary.piecewise import Piecewise
        if self.has(Piecewise):
            return None
        return hypersimp(self, k) is not None

    @property
    def is_comparable(self):
        """Return True if self can be computed to a real number
        (or already is a real number) with precision, else False.

        Examples
        ========

        >>> from sympy import exp_polar, pi, I
        >>> (I*exp_polar(I*pi/2)).is_comparable
        True
        >>> (I*exp_polar(I*pi*2)).is_comparable
        False

        A False result does not mean that `self` cannot be rewritten
        into a form that would be comparable. For example, the
        difference computed below is zero but without simplification
        it does not evaluate to a zero with precision:

        >>> e = 2**pi*(1 + 2**pi)
        >>> dif = e - e.expand()
        >>> dif.is_comparable
        False
        >>> dif.n(2)._prec
        1

        """
        return self._eval_is_comparable()

    def _eval_is_comparable(self) -> bool:
        # Expr.is_comparable overrides this
        return False

    @property
    def func(self):
        """
        The top-level function in an expression.

        The following should hold for all objects::

            >> x == x.func(*x.args)

        Examples
        ========

        >>> from sympy.abc import x
        >>> a = 2*x
        >>> a.func
        <class 'sympy.core.mul.Mul'>
        >>> a.args
        (2, x)
        >>> a.func(*a.args)
        2*x
        >>> a == a.func(*a.args)
        True

        """
        return self.__class__

    @property
    def args(self) -> tuple[Basic, ...]:
        """Returns a tuple of arguments of 'self'.

        Examples
        ========

        >>> from sympy import cot
        >>> from sympy.abc import x, y

        >>> cot(x).args
        (x,)

        >>> cot(x).args[0]
        x

        >>> (x*y).args
        (x, y)

        >>> (x*y).args[1]
        y

        Notes
        =====

        Never use self._args, always use self.args.
        Only use _args in __new__ when creating a new function.
        Do not override .args() from Basic (so that it is easy to
        change the interface in the future if needed).
        """
        return self._args

    @property
    def _sorted_args(self):
        """
        The same as ``args``.  Derived classes which do not fix an
        order on their arguments should override this method to
        produce the sorted representation.
        """
        return self.args

    def as_content_primitive(self, radical=False, clear=True):
        """A stub to allow Basic args (like Tuple) to be skipped when computing
        the content and primitive components of an expression.

        See Also
        ========

        sympy.core.expr.Expr.as_content_primitive
        """
        return S.One, self

    @overload
    def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
    @overload
    def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
    @overload
    def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

    def subs(self, arg1: Mapping[Basic | complex, Basic | complex]
            | Iterable[tuple[Basic | complex, Basic | complex]] | Basic | complex,
             arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
        """
        Substitutes old for new in an expression after sympifying args.

        `args` is either:
          - two arguments, e.g. foo.subs(old, new)
          - one iterable argument, e.g. foo.subs(iterable). The iterable may be
             o an iterable container with (old, new) pairs. In this case the
               replacements are processed in the order given with successive
               patterns possibly affecting replacements already made.
             o a dict or set whose key/value items correspond to old/new pairs.
               In this case the old/new pairs will be sorted by op count and in
               case of a tie, by number of args and the default_sort_key. The
               resulting sorted list is then processed as an iterable container
               (see previous).

        If the keyword ``simultaneous`` is True, the subexpressions will not be
        evaluated until all the substitutions have been made.

        Examples
        ========

        >>> from sympy import pi, exp, limit, oo
        >>> from sympy.abc import x, y
        >>> (1 + x*y).subs(x, pi)
        pi*y + 1
        >>> (1 + x*y).subs({x:pi, y:2})
        1 + 2*pi
        >>> (1 + x*y).subs([(x, pi), (y, 2)])
        1 + 2*pi
        >>> reps = [(y, x**2), (x, 2)]
        >>> (x + y).subs(reps)
        6
        >>> (x + y).subs(reversed(reps))
        x**2 + 2

        >>> (x**2 + x**4).subs(x**2, y)
        y**2 + y

        To replace only the x**2 but not the x**4, use xreplace:

        >>> (x**2 + x**4).xreplace({x**2: y})
        x**4 + y

        To delay evaluation until all substitutions have been made,
        set the keyword ``simultaneous`` to True:

        >>> (x/y).subs([(x, 0), (y, 0)])
        0
        >>> (x/y).subs([(x, 0), (y, 0)], simultaneous=True)
        nan

        This has the added feature of not allowing subsequent substitutions
        to affect those already made:

        >>> ((x + y)/y).subs({x + y: y, y: x + y})
        1
        >>> ((x + y)/y).subs({x + y: y, y: x + y}, simultaneous=True)
        y/(x + y)

        In order to obtain a canonical result, unordered iterables are
        sorted by count_op length, number of arguments and by the
        default_sort_key to break any ties. All other iterables are left
        unsorted.

        >>> from sympy import sqrt, sin, cos
        >>> from sympy.abc import a, b, c, d, e

        >>> A = (sqrt(sin(2*x)), a)
        >>> B = (sin(2*x), b)
        >>> C = (cos(2*x), c)
        >>> D = (x, d)
        >>> E = (exp(x), e)

        >>> expr = sqrt(sin(2*x))*sin(exp(x)*x)*cos(2*x) + sin(2*x)

        >>> expr.subs(dict([A, B, C, D, E]))
        a*c*sin(d*e) + b

        The resulting expression represents a literal replacement of the
        old arguments with the new arguments. This may not reflect the
        limiting behavior of the expression:

        >>> (x**3 - 3*x).subs({x: oo})
        nan

        >>> limit(x**3 - 3*x, x, oo)
        oo

        If the substitution will be followed by numerical
        evaluation, it is better to pass the substitution to
        evalf as

        >>> (1/x).evalf(subs={x: 3.0}, n=21)
        0.333333333333333333333

        rather than

        >>> (1/x).subs({x: 3.0}).evalf(21)
        0.333333333333333314830

        as the former will ensure that the desired level of precision is
        obtained.

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules
        sympy.core.evalf.EvalfMixin.evalf: calculates the given formula to a desired level of precision

        """
        from .containers import Dict
        from .symbol import Dummy, Symbol
        from .numbers import _illegal

        items: Iterable[tuple[Basic | complex, Basic | complex]]

        unordered = False
        if arg2 is None:

            if isinstance(arg1, set):
                items = arg1
                unordered = True
            elif isinstance(arg1, (Dict, Mapping)):
                unordered = True
                items = arg1.items() # type: ignore
            elif not iterable(arg1):
                raise ValueError(filldedent("""
                   When a single argument is passed to subs
                   it should be a dictionary of old: new pairs or an iterable
                   of (old, new) tuples."""))
            else:
                items = arg1 # type: ignore
        else:
            items = [(arg1, arg2)] # type: ignore

        def sympify_old(old) -> Basic:
            if isinstance(old, str):
                # Use Symbol rather than parse_expr for old
                return Symbol(old)
            elif isinstance(old, type):
                # Allow a type e.g. Function('f') or sin
                return sympify(old, strict=False)
            else:
                return sympify(old, strict=True)

        def sympify_new(new) -> Basic:
            if isinstance(new, (str, type)):
                # Allow a type or parse a string input
                return sympify(new, strict=False)
            else:
                return sympify(new, strict=True)

        sequence = [(sympify_old(s1), sympify_new(s2)) for s1, s2 in items]

        # skip if there is no change
        sequence = [(s1, s2) for s1, s2 in sequence if not _aresame(s1, s2)]

        simultaneous = kwargs.pop('simultaneous', False)

        if unordered:
            from .sorting import _nodes, default_sort_key
            sequence_dict = dict(sequence)
            # order so more complex items are first and items
            # of identical complexity are ordered so
            # f(x) < f(y) < x < y
            # \___ 2 __/    \_1_/  <- number of nodes
            #
            # For more complex ordering use an unordered sequence.
            k = list(ordered(sequence_dict, default=False, keys=(
                lambda x: -_nodes(x),
                default_sort_key,
                )))
            sequence = [(k, sequence_dict[k]) for k in k]
            # do infinities first
            if not simultaneous:
                redo = [i for i, seq in enumerate(sequence) if seq[1] in _illegal]
                for i in reversed(redo):
                    sequence.insert(0, sequence.pop(i))

        if simultaneous:  # XXX should this be the default for dict subs?
            reps = {}
            rv = self
            kwargs['hack2'] = True
            m = Dummy('subs_m')
            for old, new in sequence:
                com = new.is_commutative
                if com is None:
                    com = True
                d = Dummy('subs_d', commutative=com)
                # using d*m so Subs will be used on dummy variables
                # in things like Derivative(f(x, y), x) in which x
                # is both free and bound
                rv = rv._subs(old, d*m, **kwargs)
                if not isinstance(rv, Basic):
                    break
                reps[d] = new
            reps[m] = S.One  # get rid of m
            return rv.xreplace(reps)
        else:
            rv = self
            for old, new in sequence:
                rv = rv._subs(old, new, **kwargs)
                if not isinstance(rv, Basic):
                    break
            return rv

    @cacheit
    def _subs(self, old, new, **hints):
        """Substitutes an expression old -> new.

        If self is not equal to old then _eval_subs is called.
        If _eval_subs does not want to make any special replacement
        then a None is received which indicates that the fallback
        should be applied wherein a search for replacements is made
        amongst the arguments of self.

        >>> from sympy import Add
        >>> from sympy.abc import x, y, z

        Examples
        ========

        Add's _eval_subs knows how to target x + y in the following
        so it makes the change:

        >>> (x + y + z).subs(x + y, 1)
        z + 1

        Add's _eval_subs does not need to know how to find x + y in
        the following:

        >>> Add._eval_subs(z*(x + y) + 3, x + y, 1) is None
        True

        The returned None will cause the fallback routine to traverse the args and
        pass the z*(x + y) arg to Mul where the change will take place and the
        substitution will succeed:

        >>> (z*(x + y) + 3).subs(x + y, 1)
        z + 3

        ** Developers Notes **

        An _eval_subs routine for a class should be written if:

            1) any arguments are not instances of Basic (e.g. bool, tuple);

            2) some arguments should not be targeted (as in integration
               variables);

            3) if there is something other than a literal replacement
               that should be attempted (as in Piecewise where the condition
               may be updated without doing a replacement).

        If it is overridden, here are some special cases that might arise:

            1) If it turns out that no special change was made and all
               the original sub-arguments should be checked for
               replacements then None should be returned.

            2) If it is necessary to do substitutions on a portion of
               the expression then _subs should be called. _subs will
               handle the case of any sub-expression being equal to old
               (which usually would not be the case) while its fallback
               will handle the recursion into the sub-arguments. For
               example, after Add's _eval_subs removes some matching terms
               it must process the remaining terms so it calls _subs
               on each of the un-matched terms and then adds them
               onto the terms previously obtained.

           3) If the initial expression should remain unchanged then
              the original expression should be returned. (Whenever an
              expression is returned, modified or not, no further
              substitution of old -> new is attempted.) Sum's _eval_subs
              routine uses this strategy when a substitution is attempted
              on any of its summation variables.
        """

        def fallback(self, old, new):
            """
            Try to replace old with new in any of self's arguments.
            """
            hit = False
            args = list(self.args)
            for i, arg in enumerate(args):
                if not hasattr(arg, '_eval_subs'):
                    continue
                arg = arg._subs(old, new, **hints)
                if not _aresame(arg, args[i]):
                    hit = True
                    args[i] = arg
            if hit:
                rv = self.func(*args)
                hack2 = hints.get('hack2', False)
                if hack2 and self.is_Mul and not rv.is_Mul:  # 2-arg hack
                    coeff = S.One
                    nonnumber = []
                    for i in args:
                        if i.is_Number:
                            coeff *= i
                        else:
                            nonnumber.append(i)
                    nonnumber = self.func(*nonnumber)
                    if coeff is S.One:
                        return nonnumber
                    else:
                        return self.func(coeff, nonnumber, evaluate=False)
                return rv
            return self

        if _aresame(self, old):
            return new

        rv = self._eval_subs(old, new)
        if rv is None:
            rv = fallback(self, old, new)
        return rv

    def _eval_subs(self, old, new) -> Basic | None:
        """Override this stub if you want to do anything more than
        attempt a replacement of old with new in the arguments of self.

        See also
        ========

        _subs
        """
        return None

    def xreplace(self, rule):
        """
        Replace occurrences of objects within the expression.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule

        Returns
        =======

        xreplace : the result of the replacement

        Examples
        ========

        >>> from sympy import symbols, pi, exp
        >>> x, y, z = symbols('x y z')
        >>> (1 + x*y).xreplace({x: pi})
        pi*y + 1
        >>> (1 + x*y).xreplace({x: pi, y: 2})
        1 + 2*pi

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> (x*y + z).xreplace({x*y: pi})
        z + pi
        >>> (x*y*z).xreplace({x*y: pi})
        x*y*z
        >>> (2*x).xreplace({2*x: y, x: z})
        y
        >>> (2*2*x).xreplace({2*x: y, x: z})
        4*z
        >>> (x + y + 2).xreplace({x + y: 2})
        x + y + 2
        >>> (x + 2 + exp(x + 2)).xreplace({x + 2: y})
        x + exp(y) + 2

        xreplace does not differentiate between free and bound symbols. In the
        following, subs(x, y) would not change x since it is a bound symbol,
        but xreplace does:

        >>> from sympy import Integral
        >>> Integral(x, (x, 1, 2*x)).xreplace({x: y})
        Integral(y, (y, 1, 2*y))

        Trying to replace x with an expression raises an error:

        >>> Integral(x, (x, 1, 2*x)).xreplace({x: 2*y}) # doctest: +SKIP
        ValueError: Invalid limits given: ((2*y, 1, 4*y),)

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        subs: substitution of subexpressions as defined by the objects
              themselves.

        """
        value, _ = self._xreplace(rule)
        return value

    def _xreplace(self, rule):
        """
        Helper for xreplace. Tracks whether a replacement actually occurred.
        """
        if self in rule:
            return rule[self], True
        elif rule:
            args = []
            changed = False
            for a in self.args:
                _xreplace = getattr(a, '_xreplace', None)
                if _xreplace is not None:
                    a_xr = _xreplace(rule)
                    args.append(a_xr[0])
                    changed |= a_xr[1]
                else:
                    args.append(a)
            args = tuple(args)
            if changed:
                return self.func(*args), True
        return self, False

    @cacheit
    def has(self, *patterns):
        """
        Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import sin
        >>> from sympy.abc import x, y, z
        >>> (x**2 + sin(x*y)).has(z)
        False
        >>> (x**2 + sin(x*y)).has(x, y, z)
        True
        >>> x.has(x)
        True

        Note ``has`` is a structural algorithm with no knowledge of
        mathematics. Consider the following half-open interval:

        >>> from sympy import Interval
        >>> i = Interval.Lopen(0, 5); i
        Interval.Lopen(0, 5)
        >>> i.args
        (0, 5, True, False)
        >>> i.has(4)  # there is no "4" in the arguments
        False
        >>> i.has(0)  # there *is* a "0" in the arguments
        True

        Instead, use ``contains`` to determine whether a number is in the
        interval or not:

        >>> i.contains(4)
        True
        >>> i.contains(0)
        False


        Note that ``expr.has(*patterns)`` is exactly equivalent to
        ``any(expr.has(p) for p in patterns)``. In particular, ``False`` is
        returned when the list of patterns is empty.

        >>> x.has()
        False

        """
        return self._has(iterargs, *patterns)

    def has_xfree(self, s: set[Basic]):
        """Return True if self has any of the patterns in s as a
        free argument, else False. This is like `Basic.has_free`
        but this will only report exact argument matches.

        Examples
        ========

        >>> from sympy import Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> f(x).has_xfree({f})
        False
        >>> f(x).has_xfree({f(x)})
        True
        >>> f(x + 1).has_xfree({x})
        True
        >>> f(x + 1).has_xfree({x + 1})
        True
        >>> f(x + y + 1).has_xfree({x + 1})
        False
        """
        # protect O(1) containment check by requiring:
        if type(s) is not set:
            raise TypeError('expecting set argument')
        return any(a in s for a in iterfreeargs(self))

    @cacheit
    def has_free(self, *patterns):
        """Return True if self has object(s) ``x`` as a free expression
        else False.

        Examples
        ========

        >>> from sympy import Integral, Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> g = Function('g')
        >>> expr = Integral(f(x), (f(x), 1, g(y)))
        >>> expr.free_symbols
        {y}
        >>> expr.has_free(g(y))
        True
        >>> expr.has_free(*(x, f(x)))
        False

        This works for subexpressions and types, too:

        >>> expr.has_free(g)
        True
        >>> (x + y + 1).has_free(y + 1)
        True
        """
        if not patterns:
            return False
        p0 = patterns[0]
        if len(patterns) == 1 and iterable(p0) and not isinstance(p0, Basic):
            # Basic can contain iterables (though not non-Basic, ideally)
            # but don't encourage mixed passing patterns
            raise TypeError(filldedent('''
                Expecting 1 or more Basic args, not a single
                non-Basic iterable. Don't forget to unpack
                iterables: `eq.has_free(*patterns)`'''))
        # try quick test first
        s = set(patterns)
        rv = self.has_xfree(s)
        if rv:
            return rv
        # now try matching through slower _has
        return self._has(iterfreeargs, *patterns)

    def _has(self, iterargs, *patterns):
        # separate out types and unhashable objects
        type_set = set()  # only types
        p_set = set()  # hashable non-types
        for p in patterns:
            if isinstance(p, type) and issubclass(p, Basic):
                type_set.add(p)
                continue
            if not isinstance(p, Basic):
                try:
                    p = _sympify(p)
                except SympifyError:
                    continue  # Basic won't have this in it
            p_set.add(p)  # fails if object defines __eq__ but
                          # doesn't define __hash__
        types = tuple(type_set)   #
        for i in iterargs(self):  #
            if i in p_set:        # <--- here, too
                return True
            if isinstance(i, types):
                return True

        # use matcher if defined, e.g. operations defines
        # matcher that checks for exact subset containment,
        # (x + y + 1).has(x + 1) -> True
        for i in p_set - type_set:  # types don't have matchers
            if not hasattr(i, '_has_matcher'):
                continue
            match = i._has_matcher()
            if any(match(arg) for arg in iterargs(self)):
                return True

        # no success
        return False

    def replace(self, query, value, map=False, simultaneous=True, exact=None) -> Basic:
        """
        Replace matching subexpressions of ``self`` with ``value``.

        If ``map = True`` then also return the mapping {old: new} where ``old``
        was a sub-expression found with query and ``new`` is the replacement
        value for it. If the expression itself does not match the query, then
        the returned value will be ``self.xreplace(map)`` otherwise it should
        be ``self.subs(ordered(map.items()))``.

        Traverses an expression tree and performs replacement of matching
        subexpressions from the bottom to the top of the tree. The default
        approach is to do the replacement in a simultaneous fashion so
        changes made are targeted only once. If this is not desired or causes
        problems, ``simultaneous`` can be set to False.

        In addition, if an expression containing more than one Wild symbol
        is being used to match subexpressions and the ``exact`` flag is None
        it will be set to True so the match will only succeed if all non-zero
        values are received for each Wild that appears in the match pattern.
        Setting this to False accepts a match of 0; while setting it True
        accepts all matches that have a 0 in them. See example below for
        cautions.

        The list of possible combinations of queries and replacement values
        is listed below:

        Examples
        ========

        Initial setup

        >>> from sympy import log, sin, cos, tan, Wild, Mul, Add
        >>> from sympy.abc import x, y
        >>> f = log(sin(x)) + tan(sin(x**2))

        1.1. type -> type
            obj.replace(type, newtype)

            When object of type ``type`` is found, replace it with the
            result of passing its argument(s) to ``newtype``.

            >>> f.replace(sin, cos)
            log(cos(x)) + tan(cos(x**2))
            >>> sin(x).replace(sin, cos, map=True)
            (cos(x), {sin(x): cos(x)})
            >>> (x*y).replace(Mul, Add)
            x + y

        1.2. type -> func
            obj.replace(type, func)

            When object of type ``type`` is found, apply ``func`` to its
            argument(s). ``func`` must be written to handle the number
            of arguments of ``type``.

            >>> f.replace(sin, lambda arg: sin(2*arg))
            log(sin(2*x)) + tan(sin(2*x**2))
            >>> (x*y).replace(Mul, lambda *args: sin(2*Mul(*args)))
            sin(2*x*y)

        2.1. pattern -> expr
            obj.replace(pattern(wild), expr(wild))

            Replace subexpressions matching ``pattern`` with the expression
            written in terms of the Wild symbols in ``pattern``.

            >>> a, b = map(Wild, 'ab')
            >>> f.replace(sin(a), tan(a))
            log(tan(x)) + tan(tan(x**2))
            >>> f.replace(sin(a), tan(a/2))
            log(tan(x/2)) + tan(tan(x**2/2))
            >>> f.replace(sin(a), a)
            log(x) + tan(x**2)
            >>> (x*y).replace(a*x, a)
            y

            Matching is exact by default when more than one Wild symbol
            is used: matching fails unless the match gives non-zero
            values for all Wild symbols:

            >>> (2*x + y).replace(a*x + b, b - a)
            y - 2
            >>> (2*x).replace(a*x + b, b - a)
            2*x

            When set to False, the results may be non-intuitive:

            >>> (2*x).replace(a*x + b, b - a, exact=False)
            2/x

        2.2. pattern -> func
            obj.replace(pattern(wild), lambda wild: expr(wild))

            All behavior is the same as in 2.1 but now a function in terms of
            pattern variables is used rather than an expression:

            >>> f.replace(sin(a), lambda a: sin(2*a))
            log(sin(2*x)) + tan(sin(2*x**2))

        3.1. func -> func
            obj.replace(filter, func)

            Replace subexpression ``e`` with ``func(e)`` if ``filter(e)``
            is True.

            >>> g = 2*sin(x**3)
            >>> g.replace(lambda expr: expr.is_Number, lambda expr: expr**2)
            4*sin(x**9)

        The expression itself is also targeted by the query but is done in
        such a fashion that changes are not made twice.

            >>> e = x*(x*y + 1)
            >>> e.replace(lambda x: x.is_Mul, lambda x: 2*x)
            2*x*(2*x*y + 1)

        When matching a single symbol, `exact` will default to True, but
        this may or may not be the behavior that is desired:

        Here, we want `exact=False`:

        >>> from sympy import Function
        >>> f = Function('f')
        >>> e = f(1) + f(0)
        >>> q = f(a), lambda a: f(a + 1)
        >>> e.replace(*q, exact=False)
        f(1) + f(2)
        >>> e.replace(*q, exact=True)
        f(0) + f(2)

        But here, the nature of matching makes selecting
        the right setting tricky:

        >>> e = x**(1 + y)
        >>> (x**(1 + y)).replace(x**(1 + a), lambda a: x**-a, exact=False)
        x
        >>> (x**(1 + y)).replace(x**(1 + a), lambda a: x**-a, exact=True)
        x**(-x - y + 1)
        >>> (x**y).replace(x**(1 + a), lambda a: x**-a, exact=False)
        x
        >>> (x**y).replace(x**(1 + a), lambda a: x**-a, exact=True)
        x**(1 - y)

        It is probably better to use a different form of the query
        that describes the target expression more precisely:

        >>> (1 + x**(1 + y)).replace(
        ... lambda x: x.is_Pow and x.exp.is_Add and x.exp.args[0] == 1,
        ... lambda x: x.base**(1 - (x.exp - 1)))
        ...
        x**(1 - y) + 1

        See Also
        ========

        subs: substitution of subexpressions as defined by the objects
              themselves.
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules

        """

        try:
            query = _sympify(query)
        except SympifyError:
            pass
        try:
            value = _sympify(value)
        except SympifyError:
            pass
        if isinstance(query, type):
            _query = lambda expr: isinstance(expr, query)

            if isinstance(value, type):
                _value = lambda expr, result: value(*expr.args)
            elif callable(value):
                _value = lambda expr, result: value(*expr.args)
            else:
                raise TypeError(
                    "given a type, replace() expects another "
                    "type or a callable")
        elif isinstance(query, Basic):
            _query = lambda expr: expr.match(query)
            if exact is None:
                from .symbol import Wild
                exact = (len(query.atoms(Wild)) > 1)

            if isinstance(value, Basic):
                if exact:
                    _value = lambda expr, result: (value.subs(result)
                        if all(result.values()) else expr)
                else:
                    _value = lambda expr, result: value.subs(result)
            elif callable(value):
                # match dictionary keys get the trailing underscore stripped
                # from them and are then passed as keywords to the callable;
                # if ``exact`` is True, only accept match if there are no null
                # values amongst those matched.
                if exact:
                    _value = lambda expr, result: (value(**
                        {str(k)[:-1]: v for k, v in result.items()})
                        if all(val for val in result.values()) else expr)
                else:
                    _value = lambda expr, result: value(**
                        {str(k)[:-1]: v for k, v in result.items()})
            else:
                raise TypeError(
                    "given an expression, replace() expects "
                    "another expression or a callable")
        elif callable(query):
            _query = query

            if callable(value):
                _value = lambda expr, result: value(expr)
            else:
                raise TypeError(
                    "given a callable, replace() expects "
                    "another callable")
        else:
            raise TypeError(
                "first argument to replace() must be a "
                "type, an expression or a callable")

        def walk(rv, F):
            """Apply ``F`` to args and then to result.
            """
            args = getattr(rv, 'args', None)
            if args is not None:
                if args:
                    newargs = tuple([walk(a, F) for a in args])
                    if args != newargs:
                        rv = rv.func(*newargs)
                        if simultaneous:
                            # if rv is something that was already
                            # matched (that was changed) then skip
                            # applying F again
                            for i, e in enumerate(args):
                                if rv == e and e != newargs[i]:
                                    return rv
                rv = F(rv)
            return rv

        mapping = {}  # changes that took place

        def rec_replace(expr):
            result = _query(expr)
            if result or result == {}:
                v = _value(expr, result)
                if v is not None and v != expr:
                    if map:
                        mapping[expr] = v
                    expr = v
            return expr

        rv = walk(self, rec_replace)
        return (rv, mapping) if map else rv # type: ignore

    def find(self, query, group=False):
        """Find all subexpressions matching a query."""
        query = _make_find_query(query)
        results = list(filter(query, _preorder_traversal(self)))

        if not group:
            return set(results)
        else:
            groups = {}

            for result in results:
                if result in groups:
                    groups[result] += 1
                else:
                    groups[result] = 1

            return groups

    def count(self, query):
        """Count the number of matching subexpressions."""
        query = _make_find_query(query)
        return sum(bool(query(sub)) for sub in _preorder_traversal(self))

    def matches(self, expr, repl_dict=None, old=False):
        """
        Helper method for match() that looks for a match between Wild symbols
        in self and expressions in expr.

        Examples
        ========

        >>> from sympy import symbols, Wild, Basic
        >>> a, b, c = symbols('a b c')
        >>> x = Wild('x')
        >>> Basic(a + x, x).matches(Basic(a + b, c)) is None
        True
        >>> Basic(a + x, x).matches(Basic(a + b + c, b + c))
        {x_: b + c}
        """
        expr = sympify(expr)
        if not isinstance(expr, self.__class__):
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        if self == expr:
            return repl_dict

        if len(self.args) != len(expr.args):
            return None

        d = repl_dict  # already a copy
        for arg, other_arg in zip(self.args, expr.args):
            if arg == other_arg:
                continue
            if arg.is_Relational:
                try:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
                except TypeError: # Should be InvalidComparisonError when introduced
                    d = None
            else:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
            if d is None:
                return None
        return d

    def match(self, pattern, old=False):
        """
        Pattern matching.

        Wild symbols match all.

        Return ``None`` when expression (self) does not match with pattern.
        Otherwise return a dictionary such that::

          pattern.xreplace(self.match(pattern)) == self

        Examples
        ========

        >>> from sympy import Wild, Sum
        >>> from sympy.abc import x, y
        >>> p = Wild("p")
        >>> q = Wild("q")
        >>> r = Wild("r")
        >>> e = (x+y)**(x+y)
        >>> e.match(p**p)
        {p_: x + y}
        >>> e.match(p**q)
        {p_: x + y, q_: x + y}
        >>> e = (2*x)**2
        >>> e.match(p*q**r)
        {p_: 4, q_: x, r_: 2}
        >>> (p*q**r).xreplace(e.match(p*q**r))
        4*x**2

        Since match is purely structural expressions that are equivalent up to
        bound symbols will not match:

        >>> print(Sum(x, (x, 1, 2)).match(Sum(y, (y, 1, p))))
        None

        An expression with bound symbols can be matched if the pattern uses
        a distinct ``Wild`` for each bound symbol:

        >>> Sum(x, (x, 1, 2)).match(Sum(q, (q, 1, p)))
        {p_: 2, q_: x}

        The ``old`` flag will give the old-style pattern matching where
        expressions and patterns are essentially solved to give the match. Both
        of the following give None unless ``old=True``:

        >>> (x - 2).match(p - x, old=True)
        {p_: 2*x - 2}
        >>> (2/x).match(p*x, old=True)
        {p_: 2/x**2}

        See Also
        ========

        matches: pattern.matches(expr) is the same as expr.match(pattern)
        xreplace: exact structural replacement
        replace: structural replacement with pattern matching
        Wild: symbolic placeholders for expressions in pattern matching
        """
        pattern = sympify(pattern)
        return pattern.matches(self, old=old)

    def count_ops(self, visual=False):
        """Wrapper for count_ops that returns the operation count."""
        from .function import count_ops
        return count_ops(self, visual)

    def doit(self, **hints):
        """Evaluate objects that are not evaluated by default like limits,
        integrals, sums and products. All objects of this kind will be
        evaluated recursively, unless some species were excluded via 'hints'
        or unless the 'deep' hint was set to 'False'.

        >>> from sympy import Integral
        >>> from sympy.abc import x

        >>> 2*Integral(x, x)
        2*Integral(x, x)

        >>> (2*Integral(x, x)).doit()
        x**2

        >>> (2*Integral(x, x)).doit(deep=False)
        2*Integral(x, x)

        """
        if hints.get('deep', True):
            terms = [term.doit(**hints) if isinstance(term, Basic) else term
                                         for term in self.args]
            return self.func(*terms)
        else:
            return self

    def simplify(self, **kwargs) -> Basic:
        """See the simplify function in sympy.simplify"""
        from sympy.simplify.simplify import simplify
        return simplify(self, **kwargs)

    def refine(self, assumption=True):
        """See the refine function in sympy.assumptions"""
        from sympy.assumptions.refine import refine
        return refine(self, assumption)

    def _eval_derivative_n_times(self, s, n):
        # This is the default evaluator for derivatives (as called by `diff`
        # and `Derivative`), it will attempt a loop to derive the expression
        # `n` times by calling the corresponding `_eval_derivative` method,
        # while leaving the derivative unevaluated if `n` is symbolic.  This
        # method should be overridden if the object has a closed form for its
        # symbolic n-th derivative.
        from .numbers import Integer
        if isinstance(n, (int, Integer)):
            obj = self
            for i in range(n):
                prev = obj
                obj = obj._eval_derivative(s)
                if obj is None:
                    return None
                elif obj == prev:
                    break
            return obj
        else:
            return None

    def rewrite(self, *args, deep=True, **hints):
        """
        Rewrite *self* using a defined rule.

        Rewriting transforms an expression to another, which is mathematically
        equivalent but structurally different. For example you can rewrite
        trigonometric functions as complex exponentials or combinatorial
        functions as gamma function.

        This method takes a *pattern* and a *rule* as positional arguments.
        *pattern* is optional parameter which defines the types of expressions
        that will be transformed. If it is not passed, all possible expressions
        will be rewritten. *rule* defines how the expression will be rewritten.

        Parameters
        ==========

        args : Expr
            A *rule*, or *pattern* and *rule*.
            - *pattern* is a type or an iterable of types.
            - *rule* can be any object.

        deep : bool, optional
            If ``True``, subexpressions are recursively transformed. Default is
            ``True``.

        Examples
        ========

        If *pattern* is unspecified, all possible expressions are transformed.

        >>> from sympy import cos, sin, exp, I
        >>> from sympy.abc import x
        >>> expr = cos(x) + I*sin(x)
        >>> expr.rewrite(exp)
        exp(I*x)

        Pattern can be a type or an iterable of types.

        >>> expr.rewrite(sin, exp)
        exp(I*x)/2 + cos(x) - exp(-I*x)/2
        >>> expr.rewrite([cos,], exp)
        exp(I*x)/2 + I*sin(x) + exp(-I*x)/2
        >>> expr.rewrite([cos, sin], exp)
        exp(I*x)

        Rewriting behavior can be implemented by defining ``_eval_rewrite()``
        method.

        >>> from sympy import Expr, sqrt, pi
        >>> class MySin(Expr):
        ...     def _eval_rewrite(self, rule, args, **hints):
        ...         x, = args
        ...         if rule == cos:
        ...             return cos(pi/2 - x, evaluate=False)
        ...         if rule == sqrt:
        ...             return sqrt(1 - cos(x)**2)
        >>> MySin(MySin(x)).rewrite(cos)
        cos(-cos(-x + pi/2) + pi/2)
        >>> MySin(x).rewrite(sqrt)
        sqrt(1 - cos(x)**2)

        Defining ``_eval_rewrite_as_[...]()`` method is supported for backwards
        compatibility reason. This may be removed in the future and using it is
        discouraged.

        >>> class MySin(Expr):
        ...     def _eval_rewrite_as_cos(self, *args, **hints):
        ...         x, = args
        ...         return cos(pi/2 - x, evaluate=False)
        >>> MySin(x).rewrite(cos)
        cos(-x + pi/2)

        """
        if not args:
            return self

        hints.update(deep=deep)

        pattern = args[:-1]
        rule = args[-1]

        # Special case: map `abs` to `Abs`
        if rule is abs:
            from sympy.functions.elementary.complexes import Abs
            rule = Abs

        # support old design by _eval_rewrite_as_[...] method
        if isinstance(rule, str):
            method = "_eval_rewrite_as_%s" % rule
        elif hasattr(rule, "__name__"):
            # rule is class or function
            clsname = rule.__name__
            method = "_eval_rewrite_as_%s" % clsname
        else:
            # rule is instance
            clsname = rule.__class__.__name__
            method = "_eval_rewrite_as_%s" % clsname

        if pattern:
            if iterable(pattern[0]):
                pattern = pattern[0]
            pattern = tuple(p for p in pattern if self.has(p))
            if not pattern:
                return self
        # hereafter, empty pattern is interpreted as all pattern.

        return self._rewrite(pattern, rule, method, **hints)

    def _rewrite(self, pattern, rule, method, **hints):
        deep = hints.pop('deep', True)
        if deep:
            args = [a._rewrite(pattern, rule, method, **hints)
                    for a in self.args]
        else:
            args = self.args
        if not pattern or any(isinstance(self, p) for p in pattern):
            meth = getattr(self, method, None)
            if meth is not None:
                rewritten = meth(*args, **hints)
            else:
                rewritten = self._eval_rewrite(rule, args, **hints)
            if rewritten is not None:
                return rewritten
        if not args:
            return self
        return self.func(*args)

    def _eval_rewrite(self, rule, args, **hints):
        return None

    _constructor_postprocessor_mapping = {}  # type: ignore

    @classmethod
    def _exec_constructor_postprocessors(cls, obj):
        # WARNING: This API is experimental.

        # This is an experimental API that introduces constructor
        # postprosessors for SymPy Core elements. If an argument of a SymPy
        # expression has a `_constructor_postprocessor_mapping` attribute, it will
        # be interpreted as a dictionary containing lists of postprocessing
        # functions for matching expression node names.

        clsname = obj.__class__.__name__
        postprocessors = set()
        for i in obj.args:
            for f in _get_postprocessors(clsname, type(i)):
                postprocessors.add(f)

        for f in postprocessors:
            obj = f(obj)

        return obj

    def _sage_(self):
        """
        Convert *self* to a symbolic expression of SageMath.

        This version of the method is merely a placeholder.
        """
        old_method = self._sage_
        from sage.interfaces.sympy import sympy_init # type: ignore
        sympy_init()  # may monkey-patch _sage_ method into self's class or superclasses
        if old_method == self._sage_:
            raise NotImplementedError('conversion to SageMath is not implemented')
        else:
            # call the freshly monkey-patched method
            return self._sage_()

    def could_extract_minus_sign(self) -> bool:
        return False  # see Expr.could_extract_minus_sign

    def is_same(a, b, approx=None):
        """Return True if a and b are structurally the same, else False.
        If `approx` is supplied, it will be used to test whether two
        numbers are the same or not. By default, only numbers of the
        same type will compare equal, so S.Half != Float(0.5).

        Examples
        ========

        In SymPy (unlike Python) two numbers do not compare the same if they are
        not of the same type:

        >>> from sympy import S
        >>> 2.0 == S(2)
        False
        >>> 0.5 == S.Half
        False

        By supplying a function with which to compare two numbers, such
        differences can be ignored. e.g. `equal_valued` will return True
        for decimal numbers having a denominator that is a power of 2,
        regardless of precision.

        >>> from sympy import Float
        >>> from sympy.core.numbers import equal_valued
        >>> (S.Half/4).is_same(Float(0.125, 1), equal_valued)
        True
        >>> Float(1, 2).is_same(Float(1, 10), equal_valued)
        True

        But decimals without a power of 2 denominator will compare
        as not being the same.

        >>> Float(0.1, 9).is_same(Float(0.1, 10), equal_valued)
        False

        But arbitrary differences can be ignored by supplying a function
        to test the equivalence of two numbers:

        >>> import math
        >>> Float(0.1, 9).is_same(Float(0.1, 10), math.isclose)
        True

        Other objects might compare the same even though types are not the
        same. This routine will only return True if two expressions are
        identical in terms of class types.

        >>> from sympy import eye, Basic
        >>> eye(1) == S(eye(1))  # mutable vs immutable
        True
        >>> Basic.is_same(eye(1), S(eye(1)))
        False

        """
        from .numbers import Number
        from .traversal import postorder_traversal as pot
        for t in zip_longest(pot(a), pot(b)):
            if None in t:
                return False
            a, b = t
            if isinstance(a, Number):
                if not isinstance(b, Number):
                    return False
                if approx:
                    return approx(a, b)
            if not (a == b and a.__class__ == b.__class__):
                return False
        return True


class Atom(Basic):
    """
    A parent class for atomic things. An atom is an expression with no subexpressions.

    Examples
    ========

    Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """

    is_Atom = True

    __slots__ = ()

    def matches(self, expr, repl_dict=None, old=False):
        if self == expr:
            if repl_dict is None:
                return {}
            return repl_dict.copy()

    def xreplace(self, rule, hack2=False):
        return rule.get(self, self)

    def doit(self, **hints):
        return self

    @classmethod
    def class_key(cls):
        return 2, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (str(self),)), S.One.sort_key(), S.One

    def _eval_simplify(self, **kwargs):
        return self

    @property
    def _sorted_args(self):
        # this is here as a safeguard against accidentally using _sorted_args
        # on Atoms -- they cannot be rebuilt as atom.func(*atom._sorted_args)
        # since there are no args. So the calling routine should be checking
        # to see that this property is not called for Atoms.
        raise AttributeError('Atoms have no args. It might be necessary'
        ' to make a check for Atoms in the calling code.')


@sympify_method_args
class Expr(Basic, EvalfMixin):
    """
    Base class for algebraic expressions.

    Explanation
    ===========

    Everything that requires arithmetic operations to be defined
    should subclass this class, instead of Basic (which should be
    used only for argument storage and expression manipulation, i.e.
    pattern matching, substitutions, etc).

    If you want to override the comparisons of expressions:
    Should use _eval_is_ge for inequality, or _eval_is_eq, with multiple dispatch.
    _eval_is_ge return true if x >= y, false if x < y, and None if the two types
    are not comparable or the comparison is indeterminate

    See Also
    ========

    sympy.core.basic.Basic
    """

    __slots__: tuple[str, ...] = ()

    if TYPE_CHECKING:

        def __new__(cls, *args: Basic) -> Self:
            ...

        @overload # type: ignore
        def subs(self, arg1: Mapping[Basic | complex, Expr | complex], arg2: None=None) -> Expr: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Expr | complex]], arg2: None=None, **kwargs: Any) -> Expr: ...
        @overload
        def subs(self, arg1: Expr | complex, arg2: Expr | complex) -> Expr: ...
        @overload
        def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

        def subs(self, arg1: Mapping[Basic | complex, Basic | complex] | Basic | complex, # type: ignore
                 arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
            ...

        def simplify(self, **kwargs) -> Expr:
            ...

        def evalf(self, n: int = 15, subs: dict[Basic, Basic | float] | None = None,
                  maxn: int = 100, chop: bool = False, strict: bool  = False,
                  quad: str | None = None, verbose: bool = False) -> Expr:
            ...

        n = evalf

    is_scalar = True  # self derivative is 1

    @property
    def _diff_wrt(self):
        """Return True if one can differentiate with respect to this
        object, else False.

        Explanation
        ===========

        Subclasses such as Symbol, Function and Derivative return True
        to enable derivatives wrt them. The implementation in Derivative
        separates the Symbol and non-Symbol (_diff_wrt=True) variables and
        temporarily converts the non-Symbols into Symbols when performing
        the differentiation. By default, any object deriving from Expr
        will behave like a scalar with self.diff(self) == 1. If this is
        not desired then the object must also set `is_scalar = False` or
        else define an _eval_derivative routine.

        Note, see the docstring of Derivative for how this should work
        mathematically. In particular, note that expr.subs(yourclass, Symbol)
        should be well-defined on a structural level, or this will lead to
        inconsistent results.

        Examples
        ========

        >>> from sympy import Expr
        >>> e = Expr()
        >>> e._diff_wrt
        False
        >>> class MyScalar(Expr):
        ...     _diff_wrt = True
        ...
        >>> MyScalar().diff(MyScalar())
        1
        >>> class MySymbol(Expr):
        ...     _diff_wrt = True
        ...     is_scalar = False
        ...
        >>> MySymbol().diff(MySymbol())
        Derivative(MySymbol(), MySymbol())
        """
        return False

    @cacheit
    def sort_key(self, order=None):

        coeff, expr = self.as_coeff_Mul()

        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            if base is S.Exp1:
                # If we remove this, many doctests will go crazy:
                # (keeps E**x sorted like the exp(x) function,
                #  part of exp(x) to E**x transition)
                base, exp = Function("exp")(exp), S.One
            expr = base
        else:
            exp = S.One

        if expr.is_Dummy:
            args = (expr.sort_key(),)
        elif expr.is_Atom:
            args = (str(expr),)
        else:
            if expr.is_Add:
                args = expr.as_ordered_terms(order=order)
            elif expr.is_Mul:
                args = expr.as_ordered_factors(order=order)
            else:
                args = expr.args

            args = tuple(
                [ default_sort_key(arg, order=order) for arg in args ])

        args = (len(args), tuple(args))
        exp = exp.sort_key(order=order)

        return expr.class_key(), args, exp, coeff

    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.
        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args

    # ***************
    # * Arithmetics *
    # ***************
    # Expr and its subclasses use _op_priority to determine which object
    # passed to a binary special method (__mul__, etc.) will handle the
    # operation. In general, the 'call_highest_priority' decorator will choose
    # the object with the highest _op_priority to handle the call.
    # Custom subclasses that want to define their own binary special methods
    # should set an _op_priority value that is higher than the default.
    #
    # **NOTE**:
    # This is a temporary fix, and will eventually be replaced with
    # something better and more powerful.  See issue 5510.
    _op_priority = 10.0

    @property
    def _add_handler(self):
        return Add

    @property
    def _mul_handler(self):
        return Mul

    def __pos__(self) -> Expr:
        return self

    def __neg__(self) -> Expr:
        # Mul has its own __neg__ routine, so we just
        # create a 2-args Mul with the -1 in the canonical
        # slot 0.
        c = self.is_commutative
        return Mul._from_args((S.NegativeOne, self), c)

    def __abs__(self) -> Expr:
        from sympy.functions.elementary.complexes import Abs
        return Abs(self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other) -> Expr:
        return Add(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other) -> Expr:
        return Add(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other) -> Expr:
        return Add(self, -other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other) -> Expr:
        return Add(other, -self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other) -> Expr:
        return Mul(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other) -> Expr:
        return Mul(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rpow__')
    def _pow(self, other):
        return Pow(self, other)

    def __pow__(self, other, mod=None) -> Expr:
        if mod is None:
            return self._pow(other)
        try:
            _self, other, mod = as_int(self), as_int(other), as_int(mod)
            if other >= 0:
                return _sympify(pow(_self, other, mod))
            else:
                return _sympify(mod_inverse(pow(_self, -other, mod), mod))
        except ValueError:
            power = self._pow(other)
            try:
                return power%mod
            except TypeError:
                return NotImplemented

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other) -> Expr:
        return Pow(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other) -> Expr:
        denom = Pow(other, S.NegativeOne)
        if self is S.One:
            return denom
        else:
            return Mul(self, denom)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other) -> Expr:
        denom = Pow(self, S.NegativeOne)
        if other is S.One:
            return denom
        else:
            return Mul(other, denom)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rmod__')
    def __mod__(self, other) -> Expr:
        return Mod(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__mod__')
    def __rmod__(self, other) -> Expr:
        return Mod(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rfloordiv__')
    def __floordiv__(self, other) -> Expr:
        from sympy.functions.elementary.integers import floor
        return floor(self / other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__floordiv__')
    def __rfloordiv__(self, other) -> Expr:
        from sympy.functions.elementary.integers import floor
        return floor(other / self)


    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rdivmod__')
    def __divmod__(self, other) -> tuple[Expr, Expr]:
        from sympy.functions.elementary.integers import floor
        return floor(self / other), Mod(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__divmod__')
    def __rdivmod__(self, other) -> tuple[Expr, Expr]:
        from sympy.functions.elementary.integers import floor
        return floor(other / self), Mod(other, self)

    def __int__(self) -> int:
        if not self.is_number:
            raise TypeError("Cannot convert symbols to int")
        r = self.round(2)
        if not r.is_Number:
            raise TypeError("Cannot convert complex to int")
        if r in (S.NaN, S.Infinity, S.NegativeInfinity):
            raise TypeError("Cannot convert %s to int" % r)
        i = int(r)
        if not i:
            return i
        if int_valued(r):
            # non-integer self should pass one of these tests
            if (self > i) is S.true:
                return i
            if (self < i) is S.true:
                return i - 1
            ok = self.equals(i)
            if ok is None:
                raise TypeError('cannot compute int value accurately')
            if ok:
                return i
            # off by one
            return i - (1 if i > 0 else -1)
        return i

    def __float__(self) -> float:
        # Don't bother testing if it's a number; if it's not this is going
        # to fail, and if it is we still need to check that it evalf'ed to
        # a number.
        result = self.evalf()
        if result.is_Number:
            return float(result)
        if result.is_number and result.as_real_imag()[1]:
            raise TypeError("Cannot convert complex to float")
        raise TypeError("Cannot convert expression to float")

    def __complex__(self) -> complex:
        result = self.evalf()
        re, im = result.as_real_imag()
        return complex(float(re), float(im))

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __ge__(self, other):
        from .relational import GreaterThan
        return GreaterThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __le__(self, other):
        from .relational import LessThan
        return LessThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __gt__(self, other):
        from .relational import StrictGreaterThan
        return StrictGreaterThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __lt__(self, other):
        from .relational import StrictLessThan
        return StrictLessThan(self, other)

    def __trunc__(self):
        if not self.is_number:
            raise TypeError("Cannot truncate symbols and expressions")
        else:
            return Integer(self)

    def __format__(self, format_spec: str):
        if self.is_number:
            mt = re.match(r'\+?\d*\.(\d+)f', format_spec)
            if mt:
                prec = int(mt.group(1))
                rounded = self.round(prec)
                if rounded.is_Integer:
                    return format(int(rounded), format_spec)
                if rounded.is_Float:
                    return format(rounded, format_spec)
        return super().__format__(format_spec)

    @staticmethod
    def _from_mpmath(x, prec):
        if hasattr(x, "_mpf_"):
            return Float._new(x._mpf_, prec)
        elif hasattr(x, "_mpc_"):
            re, im = x._mpc_
            re = Float._new(re, prec)
            im = Float._new(im, prec)*S.ImaginaryUnit
            return re + im
        else:
            raise TypeError("expected mpmath number (mpf or mpc)")

    @property
    def is_number(self):
        """Returns True if ``self`` has no free symbols and no
        undefined functions (AppliedUndef, to be precise). It will be
        faster than ``if not self.free_symbols``, however, since
        ``is_number`` will fail as soon as it hits a free symbol
        or undefined function.

        Examples
        ========

        >>> from sympy import Function, Integral, cos, sin, pi
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> x.is_number
        False
        >>> f(1).is_number
        False
        >>> (2*x).is_number
        False
        >>> (2 + Integral(2, x)).is_number
        False
        >>> (2 + Integral(2, (x, 1, 2))).is_number
        True

        Not all numbers are Numbers in the SymPy sense:

        >>> pi.is_number, pi.is_Number
        (True, False)

        If something is a number it should evaluate to a number with
        real and imaginary parts that are Numbers; the result may not
        be comparable, however, since the real and/or imaginary part
        of the result may not have precision.

        >>> cos(1).is_number and cos(1).is_comparable
        True

        >>> z = cos(1)**2 + sin(1)**2 - 1
        >>> z.is_number
        True
        >>> z.is_comparable
        False

        See Also
        ========

        sympy.core.basic.Basic.is_comparable
        """
        return all(obj.is_number for obj in self.args)

    def _eval_is_comparable(self):
        # Basic._eval_is_comparable always returns False, so we override it
        # here
        is_extended_real = self.is_extended_real
        if is_extended_real is False:
            return False
        if not self.is_number:
            return False

        # XXX: as_real_imag() can be a very expensive operation. It should not
        # be used here because is_comparable is used implicitly in many places.
        # Probably this method should just return self.evalf(2).is_Number.

        n, i = self.as_real_imag()

        if not n.is_Number:
            n = n.evalf(2)
            if not n.is_Number:
                return False

        if not i.is_Number:
            i = i.evalf(2)
            if not i.is_Number:
                return False

        if i:
            # if _prec = 1 we can't decide and if not,
            # the answer is False because numbers with
            # imaginary parts can't be compared
            # so return False
            return False
        else:
            return n._prec != 1

    def _random(self, n=None, re_min=-1, im_min=-1, re_max=1, im_max=1):
        """Return self evaluated, if possible, replacing free symbols with
        random complex values, if necessary.

        Explanation
        ===========

        The random complex value for each free symbol is generated
        by the random_complex_number routine giving real and imaginary
        parts in the range given by the re_min, re_max, im_min, and im_max
        values. The returned value is evaluated to a precision of n
        (if given) else the maximum of 15 and the precision needed
        to get more than 1 digit of precision. If the expression
        could not be evaluated to a number, or could not be evaluated
        to more than 1 digit of precision, then None is returned.

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.abc import x, y
        >>> x._random()                         # doctest: +SKIP
        0.0392918155679172 + 0.916050214307199*I
        >>> x._random(2)                        # doctest: +SKIP
        -0.77 - 0.87*I
        >>> (x + y/2)._random(2)                # doctest: +SKIP
        -0.57 + 0.16*I
        >>> sqrt(2)._random(2)
        1.4

        See Also
        ========

        sympy.core.random.random_complex_number
        """

        free = self.free_symbols
        prec = 1
        if free:
            from sympy.core.random import random_complex_number
            a, c, b, d = re_min, re_max, im_min, im_max
            reps = dict(list(zip(free, [random_complex_number(a, b, c, d, rational=True)
                           for zi in free])))
            try:
                nmag = abs(self.evalf(2, subs=reps))
            except (ValueError, TypeError):
                # if an out of range value resulted in evalf problems
                # then return None -- XXX is there a way to know how to
                # select a good random number for a given expression?
                # e.g. when calculating n! negative values for n should not
                # be used
                return None
        else:
            reps = {}
            nmag = abs(self.evalf(2))

        if not hasattr(nmag, '_prec'):
            # e.g. exp_polar(2*I*pi) doesn't evaluate but is_number is True
            return None

        if nmag._prec == 1:
            # increase the precision up to the default maximum
            # precision to see if we can get any significance

            # evaluate
            for prec in giant_steps(2, DEFAULT_MAXPREC):
                nmag = abs(self.evalf(prec, subs=reps))
                if nmag._prec != 1:
                    break

        if nmag._prec != 1:
            if n is None:
                n = max(prec, 15)
            return self.evalf(n, subs=reps)

        # never got any significance
        return None

    def is_constant(self, *wrt, **flags):
        """Return True if self is constant, False if not, or None if
        the constancy could not be determined conclusively.

        Explanation
        ===========

        If an expression has no free symbols then it is a constant. If
        there are free symbols it is possible that the expression is a
        constant, perhaps (but not necessarily) zero. To test such
        expressions, a few strategies are tried:

        1) numerical evaluation at two random points. If two such evaluations
        give two different values and the values have a precision greater than
        1 then self is not constant. If the evaluations agree or could not be
        obtained with any precision, no decision is made. The numerical testing
        is done only if ``wrt`` is different than the free symbols.

        2) differentiation with respect to variables in 'wrt' (or all free
        symbols if omitted) to see if the expression is constant or not. This
        will not always lead to an expression that is zero even though an
        expression is constant (see added test in test_expr.py). If
        all derivatives are zero then self is constant with respect to the
        given symbols.

        3) finding out zeros of denominator expression with free_symbols.
        It will not be constant if there are zeros. It gives more negative
        answers for expression that are not constant.

        If neither evaluation nor differentiation can prove the expression is
        constant, None is returned unless two numerical values happened to be
        the same and the flag ``failing_number`` is True -- in that case the
        numerical value will be returned.

        If flag simplify=False is passed, self will not be simplified;
        the default is True since self should be simplified before testing.

        Examples
        ========

        >>> from sympy import cos, sin, Sum, S, pi
        >>> from sympy.abc import a, n, x, y
        >>> x.is_constant()
        False
        >>> S(2).is_constant()
        True
        >>> Sum(x, (x, 1, 10)).is_constant()
        True
        >>> Sum(x, (x, 1, n)).is_constant()
        False
        >>> Sum(x, (x, 1, n)).is_constant(y)
        True
        >>> Sum(x, (x, 1, n)).is_constant(n)
        False
        >>> Sum(x, (x, 1, n)).is_constant(x)
        True
        >>> eq = a*cos(x)**2 + a*sin(x)**2 - a
        >>> eq.is_constant()
        True
        >>> eq.subs({x: pi, a: 2}) == eq.subs({x: pi, a: 3}) == 0
        True

        >>> (0**x).is_constant()
        False
        >>> x.is_constant()
        False
        >>> (x**x).is_constant()
        False
        >>> one = cos(x)**2 + sin(x)**2
        >>> one.is_constant()
        True
        >>> ((one - 1)**(x + 1)).is_constant() in (True, False) # could be 0 or 1
        True
        """

        simplify = flags.get('simplify', True)

        if self.is_number:
            return True
        free = self.free_symbols
        if not free:
            return True  # assume f(1) is some constant

        # if we are only interested in some symbols and they are not in the
        # free symbols then this expression is constant wrt those symbols
        wrt = set(wrt)
        if wrt and not wrt & free:
            return True
        wrt = wrt or free

        # simplify unless this has already been done
        expr = self
        if simplify:
            expr = expr.simplify()

        # is_zero should be a quick assumptions check; it can be wrong for
        # numbers (see test_is_not_constant test), giving False when it
        # shouldn't, but hopefully it will never give True unless it is sure.
        if expr.is_zero:
            return True

        # Don't attempt substitution or differentiation with non-number symbols
        wrt_number = {sym for sym in wrt if sym.kind is NumberKind}

        # try numerical evaluation to see if we get two different values
        failing_number = None
        if wrt_number == free:
            # try 0 (for a) and 1 (for b)
            try:
                a = expr.subs(list(zip(free, [0]*len(free))),
                    simultaneous=True)
                if a is S.NaN:
                    # evaluation may succeed when substitution fails
                    a = expr._random(None, 0, 0, 0, 0)
            except ZeroDivisionError:
                a = None
            if a is not None and a is not S.NaN:
                try:
                    b = expr.subs(list(zip(free, [1]*len(free))),
                        simultaneous=True)
                    if b is S.NaN:
                        # evaluation may succeed when substitution fails
                        b = expr._random(None, 1, 0, 1, 0)
                except ZeroDivisionError:
                    b = None
                if b is not None and b is not S.NaN and b.equals(a) is False:
                    return False
                # try random real
                b = expr._random(None, -1, 0, 1, 0)
                if b is not None and b is not S.NaN and b.equals(a) is False:
                    return False
                # try random complex
                b = expr._random()
                if b is not None and b is not S.NaN:
                    if b.equals(a) is False:
                        return False
                    failing_number = a if a.is_number else b

        # now we will test each wrt symbol (or all free symbols) to see if the
        # expression depends on them or not using differentiation. This is
        # not sufficient for all expressions, however, so we don't return
        # False if we get a derivative other than 0 with free symbols.
        for w in wrt_number:
            deriv = expr.diff(w)
            if simplify:
                deriv = deriv.simplify()
            if deriv != 0:
                if not (pure_complex(deriv, or_real=True)):
                    if flags.get('failing_number', False):
                        return failing_number
                return False
        from sympy.solvers.solvers import denoms
        return fuzzy_not(fuzzy_or(den.is_zero for den in denoms(self)))

    def equals(self, other, failing_expression=False):
        """Return True if self == other, False if it does not, or None. If
        failing_expression is True then the expression which did not simplify
        to a 0 will be returned instead of None.

        Explanation
        ===========

        If ``self`` is a Number (or complex number) that is not zero, then
        the result is False.

        If ``self`` is a number and has not evaluated to zero, evalf will be
        used to test whether the expression evaluates to zero. If it does so
        and the result has significance (i.e. the precision is either -1, for
        a Rational result, or is greater than 1) then the evalf value will be
        used to return True or False.

        """
        from sympy.simplify.simplify import nsimplify, simplify
        from sympy.solvers.solvers import solve
        from sympy.polys.polyerrors import NotAlgebraic
        from sympy.polys.numberfields import minimal_polynomial

        other = sympify(other)

        if not isinstance(other, Expr):
            return False

        if self == other:
            return True

        # they aren't the same so see if we can make the difference 0;
        # don't worry about doing simplification steps one at a time
        # because if the expression ever goes to 0 then the subsequent
        # simplification steps that are done will be very fast.
        diff = factor_terms(simplify(self - other), radical=True)

        if not diff:
            return True

        if not diff.has(Add, Mod):
            # if there is no expanding to be done after simplifying
            # then this can't be a zero
            return False

        factors = diff.as_coeff_mul()[1]
        if len(factors) > 1:  # avoid infinity recursion
            fac_zero = [fac.equals(0) for fac in factors]
            if None not in fac_zero:  # every part can be decided
                return any(fac_zero)

        constant = diff.is_constant(simplify=False, failing_number=True)

        if constant is False:
            return False

        if not diff.is_number:
            if constant is None:
                # e.g. unless the right simplification is done, a symbolic
                # zero is possible (see expression of issue 6829: without
                # simplification constant will be None).
                return

        if constant is True:
            # this gives a number whether there are free symbols or not
            ndiff = diff._random()
            # is_comparable will work whether the result is real
            # or complex; it could be None, however.
            if ndiff and ndiff.is_comparable:
                return False

        # sometimes we can use a simplified result to give a clue as to
        # what the expression should be; if the expression is *not* zero
        # then we should have been able to compute that and so now
        # we can just consider the cases where the approximation appears
        # to be zero -- we try to prove it via minimal_polynomial.
        #
        # removed
        # ns = nsimplify(diff)
        # if diff.is_number and (not ns or ns == diff):
        #
        # The thought was that if it nsimplifies to 0 that's a sure sign
        # to try the following to prove it; or if it changed but wasn't
        # zero that might be a sign that it's not going to be easy to
        # prove. But tests seem to be working without that logic.
        #
        if diff.is_number:
            # try to prove via self-consistency
            surds = [s for s in diff.atoms(Pow) if s.args[0].is_Integer]
            # it seems to work better to try big ones first
            surds.sort(key=lambda x: -x.args[0])
            for s in surds:
                try:
                    # simplify is False here -- this expression has already
                    # been identified as being hard to identify as zero;
                    # we will handle the checking ourselves using nsimplify
                    # to see if we are in the right ballpark or not and if so
                    # *then* the simplification will be attempted.
                    sol = solve(diff, s, simplify=False)
                    if sol:
                        if s in sol:
                            # the self-consistent result is present
                            return True
                        if all(si.is_Integer for si in sol):
                            # perfect powers are removed at instantiation
                            # so surd s cannot be an integer
                            return False
                        if all(i.is_algebraic is False for i in sol):
                            # a surd is algebraic
                            return False
                        if any(si in surds for si in sol):
                            # it wasn't equal to s but it is in surds
                            # and different surds are not equal
                            return False
                        if any(nsimplify(s - si) == 0 and
                                simplify(s - si) == 0 for si in sol):
                            return True
                        if s.is_real:
                            if any(nsimplify(si, [s]) == s and simplify(si) == s
                                    for si in sol):
                                return True
                except NotImplementedError:
                    pass

            # try to prove with minimal_polynomial but know when
            # *not* to use this or else it can take a long time. e.g. issue 8354
            if True:  # change True to condition that assures non-hang
                try:
                    mp = minimal_polynomial(diff)
                    if mp.is_Symbol:
                        return True
                    return False
                except (NotAlgebraic, NotImplementedError):
                    pass

        # diff has not simplified to zero; constant is either None, True
        # or the number with significance (is_comparable) that was randomly
        # calculated twice as the same value.
        if constant not in (True, None) and constant != 0:
            return False

        if failing_expression:
            return diff
        return None

    def _eval_is_extended_positive_negative(self, positive):
        from sympy.polys.numberfields import minimal_polynomial
        from sympy.polys.polyerrors import NotAlgebraic
        if self.is_number:
            # check to see that we can get a value
            try:
                n2 = self._eval_evalf(2)
            # XXX: This shouldn't be caught here
            # Catches ValueError: hypsum() failed to converge to the requested
            # 34 bits of accuracy
            except ValueError:
                return None
            if n2 is None:
                return None
            if getattr(n2, '_prec', 1) == 1:  # no significance
                return None
            if n2 is S.NaN:
                return None

            f = self.evalf(2)
            if f.is_Float:
                match = f, S.Zero
            else:
                match = pure_complex(f)
            if match is None:
                return False
            r, i = match
            if not (i.is_Number and r.is_Number):
                return False
            if r._prec != 1 and i._prec != 1:
                return bool(not i and ((r > 0) if positive else (r < 0)))
            elif r._prec == 1 and (not i or i._prec == 1) and \
                    self._eval_is_algebraic() and not self.has(Function):
                try:
                    if minimal_polynomial(self).is_Symbol:
                        return False
                except (NotAlgebraic, NotImplementedError):
                    pass

    def _eval_is_extended_positive(self):
        return self._eval_is_extended_positive_negative(positive=True)

    def _eval_is_extended_negative(self):
        return self._eval_is_extended_positive_negative(positive=False)

    def _eval_interval(self, x, a, b):
        """
        Returns evaluation over an interval.  For most functions this is:

        self.subs(x, b) - self.subs(x, a),

        possibly using limit() if NaN is returned from subs, or if
        singularities are found between a and b.

        If b or a is None, it only evaluates -self.subs(x, a) or self.subs(b, x),
        respectively.

        """
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.exponential import log
        from sympy.series.limits import limit, Limit
        from sympy.sets.sets import Interval
        from sympy.solvers.solveset import solveset

        if (a is None and b is None):
            raise ValueError('Both interval ends cannot be None.')

        def _eval_endpoint(left):
            c = a if left else b
            if c is None:
                return S.Zero
            else:
                C = self.subs(x, c)
                if C.has(S.NaN, S.Infinity, S.NegativeInfinity,
                         S.ComplexInfinity, AccumBounds):
                    if (a < b) != False:
                        C = limit(self, x, c, "+" if left else "-")
                    else:
                        C = limit(self, x, c, "-" if left else "+")

                    if isinstance(C, Limit):
                        raise NotImplementedError("Could not compute limit")
            return C

        if a == b:
            return S.Zero

        A = _eval_endpoint(left=True)
        if A is S.NaN:
            return A

        B = _eval_endpoint(left=False)

        if (a and b) is None:
            return B - A

        value = B - A

        if a.is_comparable and b.is_comparable:
            if a < b:
                domain = Interval(a, b)
            else:
                domain = Interval(b, a)
            # check the singularities of self within the interval
            # if singularities is a ConditionSet (not iterable), catch the exception and pass
            singularities = solveset(self.cancel().as_numer_denom()[1], x,
                domain=domain)
            for logterm in self.atoms(log):
                singularities = singularities | solveset(logterm.args[0], x,
                    domain=domain)
            try:
                for s in singularities:
                    if value is S.NaN:
                        # no need to keep adding, it will stay NaN
                        break
                    if not s.is_comparable:
                        continue
                    if (a < s) == (s < b) == True:
                        value += -limit(self, x, s, "+") + limit(self, x, s, "-")
                    elif (b < s) == (s < a) == True:
                        value += limit(self, x, s, "+") - limit(self, x, s, "-")
            except TypeError:
                pass

        return value

    def _eval_power(self, expt) -> Expr | None:
        # subclass to compute self**other for cases when
        # other is not NaN, 0, or 1
        return None

    def _eval_conjugate(self):
        if self.is_extended_real:
            return self
        elif self.is_imaginary:
            return -self

    def conjugate(self):
        """Returns the complex conjugate of 'self'."""
        from sympy.functions.elementary.complexes import conjugate as c
        return c(self)

    def dir(self, x, cdir):
        if self.is_zero:
            return S.Zero
        from sympy.functions.elementary.exponential import log
        minexp = S.Zero
        arg = self
        while arg:
            minexp += S.One
            arg = arg.diff(x)
            coeff = arg.subs(x, 0)
            if coeff is S.NaN:
                coeff = arg.limit(x, 0)
            if coeff is S.ComplexInfinity:
                try:
                    coeff, _ = arg.leadterm(x)
                    if coeff.has(log(x)):
                        raise ValueError()
                except ValueError:
                    coeff = arg.limit(x, 0)
            if coeff != S.Zero:
                break
        return coeff*cdir**minexp

    def _eval_transpose(self):
        from sympy.functions.elementary.complexes import conjugate
        if (self.is_complex or self.is_infinite):
            return self
        elif self.is_hermitian:
            return conjugate(self)
        elif self.is_antihermitian:
            return -conjugate(self)

    def transpose(self):
        from sympy.functions.elementary.complexes import transpose
        return transpose(self)

    def _eval_adjoint(self):
        from sympy.functions.elementary.complexes import conjugate, transpose
        if self.is_hermitian:
            return self
        elif self.is_antihermitian:
            return -self
        obj = self._eval_conjugate()
        if obj is not None:
            return transpose(obj)
        obj = self._eval_transpose()
        if obj is not None:
            return conjugate(obj)

    def adjoint(self):
        from sympy.functions.elementary.complexes import adjoint
        return adjoint(self)

    @classmethod
    def _parse_order(cls, order):
        """Parse and configure the ordering of terms. """
        from sympy.polys.orderings import monomial_key

        startswith = getattr(order, "startswith", None)
        if startswith is None:
            reverse = False
        else:
            reverse = startswith('rev-')
            if reverse:
                order = order[4:]

        monom_key = monomial_key(order)

        def neg(monom):
            return tuple([neg(m) if isinstance(m, tuple) else -m for m in monom])

        def key(term):
            _, ((re, im), monom, ncpart) = term

            monom = neg(monom_key(monom))
            ncpart = tuple([e.sort_key(order=order) for e in ncpart])
            coeff = ((bool(im), im), (re, im))

            return monom, ncpart, coeff

        return key, reverse

    def as_ordered_factors(self, order=None):
        """Return list of ordered factors (if Mul) else [self]."""
        return [self]

    def as_poly(self, *gens, **args):
        """Converts ``self`` to a polynomial or returns ``None``.

        Explanation
        ===========

        >>> from sympy import sin
        >>> from sympy.abc import x, y

        >>> print((x**2 + x*y).as_poly())
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + x*y).as_poly(x, y))
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + sin(y)).as_poly(x, y))
        None

        """
        from sympy.polys.polyerrors import PolynomialError, GeneratorsNeeded
        from sympy.polys.polytools import Poly

        try:
            poly = Poly(self, *gens, **args)

            if not poly.is_Poly:
                return None
            else:
                return poly
        except (PolynomialError, GeneratorsNeeded):
            # PolynomialError is caught for e.g. exp(x).as_poly(x)
            # GeneratorsNeeded is caught for e.g. S(2).as_poly()
            return None

    def as_ordered_terms(self, order=None, data=False):
        """
        Transform an expression to an ordered list of terms.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x

        >>> (sin(x)**2*cos(x) + sin(x)**2 + 1).as_ordered_terms()
        [sin(x)**2*cos(x), sin(x)**2, 1]

        """

        from .numbers import Number, NumberSymbol

        if order is None and self.is_Add:
            # Spot the special case of Add(Number, Mul(Number, expr)) with the
            # first number positive and the second number negative
            key = lambda x:not isinstance(x, (Number, NumberSymbol))
            add_args = sorted(Add.make_args(self), key=key)
            if (len(add_args) == 2
                and isinstance(add_args[0], (Number, NumberSymbol))
                and isinstance(add_args[1], Mul)):
                mul_args = sorted(Mul.make_args(add_args[1]), key=key)
                if (len(mul_args) == 2
                    and isinstance(mul_args[0], Number)
                    and add_args[0].is_positive
                    and mul_args[0].is_negative):
                    return add_args

        key, reverse = self._parse_order(order)
        terms, gens = self.as_terms()

        if not any(term.is_Order for term, _ in terms):
            ordered = sorted(terms, key=key, reverse=reverse)
        else:
            _terms, _order = [], []

            for term, repr in terms:
                if not term.is_Order:
                    _terms.append((term, repr))
                else:
                    _order.append((term, repr))

            ordered = sorted(_terms, key=key, reverse=True) \
                + sorted(_order, key=key, reverse=True)

        if data:
            return ordered, gens
        else:
            return [term for term, _ in ordered]

    def as_terms(self):
        """Transform an expression to a list of terms. """
        from .exprtools import decompose_power

        gens, terms = set(), []

        for term in Add.make_args(self):
            coeff, _term = term.as_coeff_Mul()

            coeff = complex(coeff)
            cpart, ncpart = {}, []

            if _term is not S.One:
                for factor in Mul.make_args(_term):
                    if factor.is_number:
                        try:
                            coeff *= complex(factor)
                        except (TypeError, ValueError):
                            pass
                        else:
                            continue

                    if factor.is_commutative:
                        base, exp = decompose_power(factor)

                        cpart[base] = exp
                        gens.add(base)
                    else:
                        ncpart.append(factor)

            coeff = coeff.real, coeff.imag
            ncpart = tuple(ncpart)

            terms.append((term, (coeff, cpart, ncpart)))

        gens = sorted(gens, key=default_sort_key)

        k, indices = len(gens), {}

        for i, g in enumerate(gens):
            indices[g] = i

        result = []

        for term, (coeff, cpart, ncpart) in terms:
            monom = [0]*k

            for base, exp in cpart.items():
                monom[indices[base]] = exp

            result.append((term, (coeff, tuple(monom), ncpart)))

        return result, gens

    def removeO(self) -> Expr:
        """Removes the additive O(..) symbol if there is one"""
        return self

    def getO(self) -> Expr | None:
        """Returns the additive O(..) symbol if there is one, else None."""
        return None

    def getn(self):
        """
        Returns the order of the expression.

        Explanation
        ===========

        The order is determined either from the O(...) term. If there
        is no O(...) term, it returns None.

        Examples
        ========

        >>> from sympy import O
        >>> from sympy.abc import x
        >>> (1 + x + O(x**2)).getn()
        2
        >>> (1 + x).getn()

        """
        o = self.getO()
        if o is None:
            return None
        elif o.is_Order:
            o = o.expr
            if o is S.One:
                return S.Zero
            if o.is_Symbol:
                return S.One
            if o.is_Pow:
                return o.args[1]
            if o.is_Mul:  # x**n*log(x)**n or x**n/log(x)**n
                for oi in o.args:
                    if oi.is_Symbol:
                        return S.One
                    if oi.is_Pow:
                        from .symbol import Dummy, Symbol
                        syms = oi.atoms(Symbol)
                        if len(syms) == 1:
                            x = syms.pop()
                            oi = oi.subs(x, Dummy('x', positive=True))
                            if oi.base.is_Symbol and oi.exp.is_Rational:
                                return abs(oi.exp)

        raise NotImplementedError('not sure of order of %s' % o)

    def count_ops(self, visual=False):
        from .function import count_ops
        return count_ops(self, visual)

    def args_cnc(self, cset=False, warn=True, split_1=True):
        """Return [commutative factors, non-commutative factors] of self.

        Explanation
        ===========

        self is treated as a Mul and the ordering of the factors is maintained.
        If ``cset`` is True the commutative factors will be returned in a set.
        If there were repeated factors (as may happen with an unevaluated Mul)
        then an error will be raised unless it is explicitly suppressed by
        setting ``warn`` to False.

        Note: -1 is always separated from a Number unless split_1 is False.

        Examples
        ========

        >>> from sympy import symbols, oo
        >>> A, B = symbols('A B', commutative=0)
        >>> x, y = symbols('x y')
        >>> (-2*x*y).args_cnc()
        [[-1, 2, x, y], []]
        >>> (-2.5*x).args_cnc()
        [[-1, 2.5, x], []]
        >>> (-2*x*A*B*y).args_cnc()
        [[-1, 2, x, y], [A, B]]
        >>> (-2*x*A*B*y).args_cnc(split_1=False)
        [[-2, x, y], [A, B]]
        >>> (-2*x*y).args_cnc(cset=True)
        [{-1, 2, x, y}, []]

        The arg is always treated as a Mul:

        >>> (-2 + x + A).args_cnc()
        [[], [x - 2 + A]]
        >>> (-oo).args_cnc() # -oo is a singleton
        [[-1, oo], []]
        """
        args = list(Mul.make_args(self))

        for i, mi in enumerate(args):
            if not mi.is_commutative:
                c = args[:i]
                nc = args[i:]
                break
        else:
            c = args
            nc = []

        if c and split_1 and (
            c[0].is_Number and
            c[0].is_extended_negative and
                c[0] is not S.NegativeOne):
            c[:1] = [S.NegativeOne, -c[0]]

        if cset:
            clen = len(c)
            c = set(c)
            if clen and warn and len(c) != clen:
                raise ValueError('repeated commutative arguments: %s' %
                                 [ci for ci in c if list(self.args).count(ci) > 1])
        return [c, nc]

    def coeff(self, x: Expr, n=1, right=False, _first=True):
        """
        Returns the coefficient from the term(s) containing ``x**n``. If ``n``
        is zero then all terms independent of ``x`` will be returned.

        Explanation
        ===========

        When ``x`` is noncommutative, the coefficient to the left (default) or
        right of ``x`` can be returned. The keyword 'right' is ignored when
        ``x`` is commutative.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.abc import x, y, z

        You can select terms that have an explicit negative in front of them:

        >>> (-x + 2*y).coeff(-1)
        x
        >>> (x - 2*y).coeff(-1)
        2*y

        You can select terms with no Rational coefficient:

        >>> (x + 2*y).coeff(1)
        x
        >>> (3 + 2*x + 4*x**2).coeff(1)
        0

        You can select terms independent of x by making n=0; in this case
        expr.as_independent(x)[0] is returned (and 0 will be returned instead
        of None):

        >>> (3 + 2*x + 4*x**2).coeff(x, 0)
        3
        >>> eq = ((x + 1)**3).expand() + 1
        >>> eq
        x**3 + 3*x**2 + 3*x + 2
        >>> [eq.coeff(x, i) for i in reversed(range(4))]
        [1, 3, 3, 2]
        >>> eq -= 2
        >>> [eq.coeff(x, i) for i in reversed(range(4))]
        [1, 3, 3, 0]

        You can select terms that have a numerical term in front of them:

        >>> (-x - 2*y).coeff(2)
        -y
        >>> from sympy import sqrt
        >>> (x + sqrt(2)*x).coeff(sqrt(2))
        x

        The matching is exact:

        >>> (3 + 2*x + 4*x**2).coeff(x)
        2
        >>> (3 + 2*x + 4*x**2).coeff(x**2)
        4
        >>> (3 + 2*x + 4*x**2).coeff(x**3)
        0
        >>> (z*(x + y)**2).coeff((x + y)**2)
        z
        >>> (z*(x + y)**2).coeff(x + y)
        0

        In addition, no factoring is done, so 1 + z*(1 + y) is not obtained
        from the following:

        >>> (x + z*(x + x*y)).coeff(x)
        1

        If such factoring is desired, factor_terms can be used first:

        >>> from sympy import factor_terms
        >>> factor_terms(x + z*(x + x*y)).coeff(x)
        z*(y + 1) + 1

        >>> n, m, o = symbols('n m o', commutative=False)
        >>> n.coeff(n)
        1
        >>> (3*n).coeff(n)
        3
        >>> (n*m + m*n*m).coeff(n) # = (1 + m)*n*m
        1 + m
        >>> (n*m + m*n*m).coeff(n, right=True) # = (1 + m)*n*m
        m

        If there is more than one possible coefficient 0 is returned:

        >>> (n*m + m*n).coeff(n)
        0

        If there is only one possible coefficient, it is returned:

        >>> (n*m + x*m*n).coeff(m*n)
        x
        >>> (n*m + x*m*n).coeff(m*n, right=1)
        1

        See Also
        ========

        as_coefficient: separate the expression into a coefficient and factor
        as_coeff_Add: separate the additive constant from an expression
        as_coeff_Mul: separate the multiplicative constant from an expression
        as_independent: separate x-dependent terms/factors from others
        sympy.polys.polytools.Poly.coeff_monomial: efficiently find the single coefficient of a monomial in Poly
        sympy.polys.polytools.Poly.nth: like coeff_monomial but powers of monomial terms are used
        """
        x = sympify(x)
        if not isinstance(x, Basic):
            return S.Zero

        n = as_int(n)

        if not x:
            return S.Zero

        if x == self:
            if n == 1:
                return S.One
            return S.Zero

        co2: list[Expr]

        if x is S.One:
            co2 = [a for a in Add.make_args(self) if a.as_coeff_Mul()[0] is S.One]
            if not co2:
                return S.Zero
            return Add(*co2)

        if n == 0:
            if x.is_Add and self.is_Add:
                c = self.coeff(x, right=right)
                if not c:
                    return S.Zero
                if not right:
                    return self - Add(*[a*x for a in Add.make_args(c)])
                return self - Add(*[x*a for a in Add.make_args(c)])
            return self.as_independent(x, as_Add=True)[0]

        # continue with the full method, looking for this power of x:
        x = x**n

        def incommon(l1, l2):
            if not l1 or not l2:
                return []
            n = min(len(l1), len(l2))
            for i in range(n):
                if l1[i] != l2[i]:
                    return l1[:i]
            return l1[:]

        def find(l, sub, first=True):
            """ Find where list sub appears in list l. When ``first`` is True
            the first occurrence from the left is returned, else the last
            occurrence is returned. Return None if sub is not in l.

            Examples
            ========

            >> l = range(5)*2
            >> find(l, [2, 3])
            2
            >> find(l, [2, 3], first=0)
            7
            >> find(l, [2, 4])
            None

            """
            if not sub or not l or len(sub) > len(l):
                return None
            n = len(sub)
            if not first:
                l.reverse()
                sub.reverse()
            for i in range(len(l) - n + 1):
                if all(l[i + j] == sub[j] for j in range(n)):
                    break
            else:
                i = None
            if not first:
                l.reverse()
                sub.reverse()
            if i is not None and not first:
                i = len(l) - (i + n)
            return i

        co2 = []
        co: list[tuple[set[Expr], list[Expr]]] = []
        args = Add.make_args(self)
        self_c = self.is_commutative
        x_c = x.is_commutative
        if self_c and not x_c:
            return S.Zero
        if _first and self.is_Add and not self_c and not x_c:
            # get the part that depends on x exactly
            xargs = Mul.make_args(x)
            d = Add(*[i for i in Add.make_args(self.as_independent(x)[1])
                if all(xi in Mul.make_args(i) for xi in xargs)])
            rv = d.coeff(x, right=right, _first=False)
            if not rv.is_Add or not right:
                return rv
            c_part, nc_part = zip(*[i.args_cnc() for i in rv.args])
            if has_variety(c_part):
                return rv
            return Add(*[Mul._from_args(i) for i in nc_part])

        one_c = self_c or x_c
        xargs, nx = x.args_cnc(cset=True, warn=bool(not x_c))
        # find the parts that pass the commutative terms
        for a in args:
            margs, nc = a.args_cnc(cset=True, warn=bool(not self_c))
            if nc is None:
                nc = []
            if len(xargs) > len(margs):
                continue
            resid = margs.difference(xargs)
            if len(resid) + len(xargs) == len(margs):
                if one_c:
                    co2.append(Mul(*(list(resid) + nc)))
                else:
                    co.append((resid, nc))
        if one_c:
            if co2 == []:
                return S.Zero
            elif co2:
                return Add(*co2)
        else:  # both nc
            # now check the non-comm parts
            if not co:
                return S.Zero
            if all(n == co[0][1] for r, n in co):
                ii = find(co[0][1], nx, right)
                if ii is not None:
                    if not right:
                        return Mul(Add(*[Mul(*r) for r, c in co]), Mul(*co[0][1][:ii]))
                    else:
                        return Mul(*co[0][1][ii + len(nx):])
            beg = reduce(incommon, (n[1] for n in co))
            if beg:
                ii = find(beg, nx, right)
                if ii is not None:
                    if not right:
                        gcdc = co[0][0]
                        for i in range(1, len(co)):
                            gcdc = gcdc.intersection(co[i][0])
                            if not gcdc:
                                break
                        return Mul(*(list(gcdc) + beg[:ii]))
                    else:
                        m = ii + len(nx)
                        return Add(*[Mul(*(list(r) + n[m:])) for r, n in co])
            end = list(reversed(
                reduce(incommon, (list(reversed(n[1])) for n in co))))
            if end:
                ii = find(end, nx, right)
                if ii is not None:
                    if not right:
                        return Add(*[Mul(*(list(r) + n[:-len(end) + ii])) for r, n in co])
                    else:
                        return Mul(*end[ii + len(nx):])
            # look for single match
            hit = None
            for i, (r, n) in enumerate(co):
                ii = find(n, nx, right)
                if ii is not None:
                    if not hit:
                        hit = ii, r, n
                    else:
                        break
            else:
                if hit:
                    ii, r, n = hit
                    if not right:
                        return Mul(*(list(r) + n[:ii]))
                    else:
                        return Mul(*n[ii + len(nx):])

            return S.Zero

    def as_expr(self, *gens):
        """
        Convert a polynomial to a SymPy expression.

        Examples
        ========

        >>> from sympy import sin
        >>> from sympy.abc import x, y

        >>> f = (x**2 + x*y).as_poly(x, y)
        >>> f.as_expr()
        x**2 + x*y

        >>> sin(x).as_expr()
        sin(x)

        """
        return self

    def as_coefficient(self, expr: Expr) -> Expr | None:
        """
        Extracts symbolic coefficient at the given expression. In
        other words, this functions separates 'self' into the product
        of 'expr' and 'expr'-free coefficient. If such separation
        is not possible it will return None.

        Examples
        ========

        >>> from sympy import E, pi, sin, I, Poly
        >>> from sympy.abc import x

        >>> E.as_coefficient(E)
        1
        >>> (2*E).as_coefficient(E)
        2
        >>> (2*sin(E)*E).as_coefficient(E)

        Two terms have E in them so a sum is returned. (If one were
        desiring the coefficient of the term exactly matching E then
        the constant from the returned expression could be selected.
        Or, for greater precision, a method of Poly can be used to
        indicate the desired term from which the coefficient is
        desired.)

        >>> (2*E + x*E).as_coefficient(E)
        x + 2
        >>> _.args[0]  # just want the exact match
        2
        >>> p = Poly(2*E + x*E); p
        Poly(x*E + 2*E, x, E, domain='ZZ')
        >>> p.coeff_monomial(E)
        2
        >>> p.nth(0, 1)
        2

        Since the following cannot be written as a product containing
        E as a factor, None is returned. (If the coefficient ``2*x`` is
        desired then the ``coeff`` method should be used.)

        >>> (2*E*x + x).as_coefficient(E)
        >>> (2*E*x + x).coeff(E)
        2*x

        >>> (E*(x + 1) + x).as_coefficient(E)

        >>> (2*pi*I).as_coefficient(pi*I)
        2
        >>> (2*I).as_coefficient(pi*I)

        See Also
        ========

        coeff: return sum of terms have a given factor
        as_coeff_Add: separate the additive constant from an expression
        as_coeff_Mul: separate the multiplicative constant from an expression
        as_independent: separate x-dependent terms/factors from others
        sympy.polys.polytools.Poly.coeff_monomial: efficiently find the single coefficient of a monomial in Poly
        sympy.polys.polytools.Poly.nth: like coeff_monomial but powers of monomial terms are used


        """

        r = self.extract_multiplicatively(expr)
        if r and not r.has(expr):
            return r
        else:
            return None

    def as_independent(self, *deps, **hint) -> tuple[Expr, Expr]:
        """
        A mostly naive separation of a Mul or Add into arguments that are not
        are dependent on deps. To obtain as complete a separation of variables
        as possible, use a separation method first, e.g.:

        * separatevars() to change Mul, Add and Pow (including exp) into Mul
        * .expand(mul=True) to change Add or Mul into Add
        * .expand(log=True) to change log expr into an Add

        The only non-naive thing that is done here is to respect noncommutative
        ordering of variables and to always return (0, 0) for `self` of zero
        regardless of hints.

        For nonzero `self`, the returned tuple (i, d) has the
        following interpretation:

        * i will has no variable that appears in deps
        * d will either have terms that contain variables that are in deps, or
          be equal to 0 (when self is an Add) or 1 (when self is a Mul)
        * if self is an Add then self = i + d
        * if self is a Mul then self = i*d
        * otherwise (self, S.One) or (S.One, self) is returned.

        To force the expression to be treated as an Add, use the hint as_Add=True

        Examples
        ========

        -- self is an Add

        >>> from sympy import sin, cos, exp
        >>> from sympy.abc import x, y, z

        >>> (x + x*y).as_independent(x)
        (0, x*y + x)
        >>> (x + x*y).as_independent(y)
        (x, x*y)
        >>> (2*x*sin(x) + y + x + z).as_independent(x)
        (y + z, 2*x*sin(x) + x)
        >>> (2*x*sin(x) + y + x + z).as_independent(x, y)
        (z, 2*x*sin(x) + x + y)

        -- self is a Mul

        >>> (x*sin(x)*cos(y)).as_independent(x)
        (cos(y), x*sin(x))

        non-commutative terms cannot always be separated out when self is a Mul

        >>> from sympy import symbols
        >>> n1, n2, n3 = symbols('n1 n2 n3', commutative=False)
        >>> (n1 + n1*n2).as_independent(n2)
        (n1, n1*n2)
        >>> (n2*n1 + n1*n2).as_independent(n2)
        (0, n1*n2 + n2*n1)
        >>> (n1*n2*n3).as_independent(n1)
        (1, n1*n2*n3)
        >>> (n1*n2*n3).as_independent(n2)
        (n1, n2*n3)
        >>> ((x-n1)*(x-y)).as_independent(x)
        (1, (x - y)*(x - n1))

        -- self is anything else:

        >>> (sin(x)).as_independent(x)
        (1, sin(x))
        >>> (sin(x)).as_independent(y)
        (sin(x), 1)
        >>> exp(x+y).as_independent(x)
        (1, exp(x + y))

        -- force self to be treated as an Add:

        >>> (3*x).as_independent(x, as_Add=True)
        (0, 3*x)

        -- force self to be treated as a Mul:

        >>> (3+x).as_independent(x, as_Add=False)
        (1, x + 3)
        >>> (-3+x).as_independent(x, as_Add=False)
        (1, x - 3)

        Note how the below differs from the above in making the
        constant on the dep term positive.

        >>> (y*(-3+x)).as_independent(x)
        (y, x - 3)

        -- use .as_independent() for true independence testing instead
           of .has(). The former considers only symbols in the free
           symbols while the latter considers all symbols

        >>> from sympy import Integral
        >>> I = Integral(x, (x, 1, 2))
        >>> I.has(x)
        True
        >>> x in I.free_symbols
        False
        >>> I.as_independent(x) == (I, 1)
        True
        >>> (I + x).as_independent(x) == (I, x)
        True

        Note: when trying to get independent terms, a separation method
        might need to be used first. In this case, it is important to keep
        track of what you send to this routine so you know how to interpret
        the returned values

        >>> from sympy import separatevars, log
        >>> separatevars(exp(x+y)).as_independent(x)
        (exp(y), exp(x))
        >>> (x + x*y).as_independent(y)
        (x, x*y)
        >>> separatevars(x + x*y).as_independent(y)
        (x, y + 1)
        >>> (x*(1 + y)).as_independent(y)
        (x, y + 1)
        >>> (x*(1 + y)).expand(mul=True).as_independent(y)
        (x, x*y)
        >>> a, b=symbols('a b', positive=True)
        >>> (log(a*b).expand(log=True)).as_independent(b)
        (log(a), log(b))

        See Also
        ========

        separatevars
        expand_log
        sympy.core.add.Add.as_two_terms
        sympy.core.mul.Mul.as_two_terms
        as_coeff_mul
        """
        from .symbol import Symbol
        from .add import _unevaluated_Add
        from .mul import _unevaluated_Mul

        if self is S.Zero:
            return (self, self)

        func = self.func
        want: type[Add] | type[Mul]
        if hint.get('as_Add', isinstance(self, Add) ):
            want = Add
        else:
            want = Mul

        # sift out deps into symbolic and other and ignore
        # all symbols but those that are in the free symbols
        sym = set()
        other = []
        for d in deps:
            if isinstance(d, Symbol):  # Symbol.is_Symbol is True
                sym.add(d)
            else:
                other.append(d)

        def has(e):
            """return the standard has() if there are no literal symbols, else
            check to see that symbol-deps are in the free symbols."""
            has_other = e.has(*other)
            if not sym:
                return has_other
            return has_other or e.has(*(e.free_symbols & sym))

        if (want is not func or
                func is not Add and func is not Mul):
            if has(self):
                return (want.identity, self)
            else:
                return (self, want.identity)
        else:
            if func is Add:
                args = list(self.args)
            else:
                args, nc = self.args_cnc()

        d = sift(args, has)
        depend = d[True]
        indep = d[False]
        if func is Add:  # all terms were treated as commutative
            return (Add(*indep), _unevaluated_Add(*depend))
        else:  # handle noncommutative by stopping at first dependent term
            for i, n in enumerate(nc):
                if has(n):
                    depend.extend(nc[i:])
                    break
                indep.append(n)
            return Mul(*indep), _unevaluated_Mul(*depend)

    def as_real_imag(self, deep=True, **hints) -> tuple[Expr, Expr]:
        """Performs complex expansion on 'self' and returns a tuple
           containing collected both real and imaginary parts. This
           method cannot be confused with re() and im() functions,
           which does not perform complex expansion at evaluation.

           However it is possible to expand both re() and im()
           functions and get exactly the same results as with
           a single call to this function.

           >>> from sympy import symbols, I

           >>> x, y = symbols('x,y', real=True)

           >>> (x + y*I).as_real_imag()
           (x, y)

           >>> from sympy.abc import z, w

           >>> (z + w*I).as_real_imag()
           (re(z) - im(w), re(w) + im(z))

        """
        if hints.get('ignore') == self:
            return None  # type: ignore
        else:
            from sympy.functions.elementary.complexes import im, re
            return (re(self), im(self))

    def as_powers_dict(self):
        """Return self as a dictionary of factors with each factor being
        treated as a power. The keys are the bases of the factors and the
        values, the corresponding exponents. The resulting dictionary should
        be used with caution if the expression is a Mul and contains non-
        commutative factors since the order that they appeared will be lost in
        the dictionary.

        See Also
        ========
        as_ordered_factors: An alternative for noncommutative applications,
                            returning an ordered list of factors.
        args_cnc: Similar to as_ordered_factors, but guarantees separation
                  of commutative and noncommutative factors.
        """
        d = defaultdict(int)
        d.update([self.as_base_exp()])
        return d

    def as_coefficients_dict(self, *syms):
        """Return a dictionary mapping terms to their Rational coefficient.
        Since the dictionary is a defaultdict, inquiries about terms which
        were not present will return a coefficient of 0.

        If symbols ``syms`` are provided, any multiplicative terms
        independent of them will be considered a coefficient and a
        regular dictionary of syms-dependent generators as keys and
        their corresponding coefficients as values will be returned.

        Examples
        ========

        >>> from sympy.abc import a, x, y
        >>> (3*x + a*x + 4).as_coefficients_dict()
        {1: 4, x: 3, a*x: 1}
        >>> _[a]
        0
        >>> (3*a*x).as_coefficients_dict()
        {a*x: 3}
        >>> (3*a*x).as_coefficients_dict(x)
        {x: 3*a}
        >>> (3*a*x).as_coefficients_dict(y)
        {1: 3*a*x}

        """
        d = defaultdict(list)
        if not syms:
            for ai in Add.make_args(self):
                c, m = ai.as_coeff_Mul()
                d[m].append(c)
            for k, v in d.items():
                if len(v) == 1:
                    d[k] = v[0]
                else:
                    d[k] = Add(*v)
        else:
            ind, dep = self.as_independent(*syms, as_Add=True)
            for i in Add.make_args(dep):
                if i.is_Mul:
                    c, x = i.as_coeff_mul(*syms)
                    if c is S.One:
                        d[i].append(c)
                    else:
                        d[i._new_rawargs(*x)].append(c)
                elif i:
                    d[i].append(S.One)
            d = {k: Add(*d[k]) for k in d}
            if ind is not S.Zero:
                d.update({S.One: ind})
        di = defaultdict(int)
        di.update(d)
        return di

    def as_base_exp(self) -> tuple[Expr, Expr]:
        # a -> b ** e
        return self, S.One

    def as_coeff_mul(self, *deps, **kwargs) -> tuple[Expr, tuple[Expr, ...]]:
        """Return the tuple (c, args) where self is written as a Mul, ``m``.

        c should be a Rational multiplied by any factors of the Mul that are
        independent of deps.

        args should be a tuple of all other factors of m; args is empty
        if self is a Number or if self is independent of deps (when given).

        This should be used when you do not know if self is a Mul or not but
        you want to treat self as a Mul or if you want to process the
        individual arguments of the tail of self as a Mul.

        - if you know self is a Mul and want only the head, use self.args[0];
        - if you do not want to process the arguments of the tail but need the
          tail then use self.as_two_terms() which gives the head and tail;
        - if you want to split self into an independent and dependent parts
          use ``self.as_independent(*deps)``

        >>> from sympy import S
        >>> from sympy.abc import x, y
        >>> (S(3)).as_coeff_mul()
        (3, ())
        >>> (3*x*y).as_coeff_mul()
        (3, (x, y))
        >>> (3*x*y).as_coeff_mul(x)
        (3*y, (x,))
        >>> (3*y).as_coeff_mul(x)
        (3*y, ())
        """
        if deps:
            if not self.has(*deps):
                return self, ()
        return S.One, (self,)

    def as_coeff_add(self, *deps) -> tuple[Expr, tuple[Expr, ...]]:
        """Return the tuple (c, args) where self is written as an Add, ``a``.

        c should be a Rational added to any terms of the Add that are
        independent of deps.

        args should be a tuple of all other terms of ``a``; args is empty
        if self is a Number or if self is independent of deps (when given).

        This should be used when you do not know if self is an Add or not but
        you want to treat self as an Add or if you want to process the
        individual arguments of the tail of self as an Add.

        - if you know self is an Add and want only the head, use self.args[0];
        - if you do not want to process the arguments of the tail but need the
          tail then use self.as_two_terms() which gives the head and tail.
        - if you want to split self into an independent and dependent parts
          use ``self.as_independent(*deps)``

        >>> from sympy import S
        >>> from sympy.abc import x, y
        >>> (S(3)).as_coeff_add()
        (3, ())
        >>> (3 + x).as_coeff_add()
        (3, (x,))
        >>> (3 + x + y).as_coeff_add(x)
        (y + 3, (x,))
        >>> (3 + y).as_coeff_add(x)
        (y + 3, ())

        """
        if deps:
            if not self.has_free(*deps):
                return self, ()
        return S.Zero, (self,)

    def primitive(self) -> tuple[Number, Expr]:
        """Return the positive Rational that can be extracted non-recursively
        from every term of self (i.e., self is treated like an Add). This is
        like the as_coeff_Mul() method but primitive always extracts a positive
        Rational (never a negative or a Float).

        Examples
        ========

        >>> from sympy.abc import x
        >>> (3*(x + 1)**2).primitive()
        (3, (x + 1)**2)
        >>> a = (6*x + 2); a.primitive()
        (2, 3*x + 1)
        >>> b = (x/2 + 3); b.primitive()
        (1/2, x + 6)
        >>> (a*b).primitive() == (1, a*b)
        True
        """
        if not self:
            return S.One, S.Zero
        c, r = self.as_coeff_Mul(rational=True)
        if c.is_negative:
            c, r = -c, -r
        return c, r

    def as_content_primitive(self, radical=False, clear=True):
        """This method should recursively remove a Rational from all arguments
        and return that (content) and the new self (primitive). The content
        should always be positive and ``Mul(*foo.as_content_primitive()) == foo``.
        The primitive need not be in canonical form and should try to preserve
        the underlying structure if possible (i.e. expand_mul should not be
        applied to self).

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.abc import x, y, z

        >>> eq = 2 + 2*x + 2*y*(3 + 3*y)

        The as_content_primitive function is recursive and retains structure:

        >>> eq.as_content_primitive()
        (2, x + 3*y*(y + 1) + 1)

        Integer powers will have Rationals extracted from the base:

        >>> ((2 + 6*x)**2).as_content_primitive()
        (4, (3*x + 1)**2)
        >>> ((2 + 6*x)**(2*y)).as_content_primitive()
        (1, (2*(3*x + 1))**(2*y))

        Terms may end up joining once their as_content_primitives are added:

        >>> ((5*(x*(1 + y)) + 2*x*(3 + 3*y))).as_content_primitive()
        (11, x*(y + 1))
        >>> ((3*(x*(1 + y)) + 2*x*(3 + 3*y))).as_content_primitive()
        (9, x*(y + 1))
        >>> ((3*(z*(1 + y)) + 2.0*x*(3 + 3*y))).as_content_primitive()
        (1, 6.0*x*(y + 1) + 3*z*(y + 1))
        >>> ((5*(x*(1 + y)) + 2*x*(3 + 3*y))**2).as_content_primitive()
        (121, x**2*(y + 1)**2)
        >>> ((x*(1 + y) + 0.4*x*(3 + 3*y))**2).as_content_primitive()
        (1, 4.84*x**2*(y + 1)**2)

        Radical content can also be factored out of the primitive:

        >>> (2*sqrt(2) + 4*sqrt(10)).as_content_primitive(radical=True)
        (2, sqrt(2)*(1 + 2*sqrt(5)))

        If clear=False (default is True) then content will not be removed
        from an Add if it can be distributed to leave one or more
        terms with integer coefficients.

        >>> (x/2 + y).as_content_primitive()
        (1/2, x + 2*y)
        >>> (x/2 + y).as_content_primitive(clear=False)
        (1, x/2 + y)
        """
        return S.One, self

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        """Return the numerator and the denominator of an expression.

        expression -> a/b -> a, b

        This is just a stub that should be defined by
        an object's class methods to get anything else.

        See Also
        ========

        normal: return ``a/b`` instead of ``(a, b)``

        """
        return self, S.One

    def normal(self):
        """Return the expression as a fraction.

        expression -> a/b

        See Also
        ========

        as_numer_denom: return ``(a, b)`` instead of ``a/b``

        """
        from .mul import _unevaluated_Mul
        n, d = self.as_numer_denom()
        if d is S.One:
            return n
        if d.is_Number:
            return _unevaluated_Mul(n, 1/d)
        else:
            return n/d

    def extract_multiplicatively(self, c: Expr) -> Expr | None:
        """Return None if it's not possible to make self in the form
           c * something in a nice way, i.e. preserving the properties
           of arguments of self.

           Examples
           ========

           >>> from sympy import symbols, Rational

           >>> x, y = symbols('x,y', real=True)

           >>> ((x*y)**3).extract_multiplicatively(x**2 * y)
           x*y**2

           >>> ((x*y)**3).extract_multiplicatively(x**4 * y)

           >>> (2*x).extract_multiplicatively(2)
           x

           >>> (2*x).extract_multiplicatively(3)

           >>> (Rational(1, 2)*x).extract_multiplicatively(3)
           x/6

        """
        from sympy.functions.elementary.exponential import exp
        from .add import _unevaluated_Add
        c = sympify(c)
        if self is S.NaN:
            return None
        if c is S.One:
            return self
        elif c == self:
            return S.One

        if c.is_Add:
            cc, pc = c.primitive()
            if cc is not S.One:
                c = Mul(cc, pc, evaluate=False)

        if c.is_Mul:
            a, b = c.as_two_terms() # type: ignore
            x = self.extract_multiplicatively(a)
            if x is not None:
                return x.extract_multiplicatively(b)
            else:
                return x

        quotient = self / c
        if self.is_Number:
            if self is S.Infinity:
                if c.is_positive:
                    return S.Infinity
            elif self is S.NegativeInfinity:
                if c.is_negative:
                    return S.Infinity
                elif c.is_positive:
                    return S.NegativeInfinity
            elif self is S.ComplexInfinity:
                if not c.is_zero:
                    return S.ComplexInfinity
            elif self.is_Integer:
                if not quotient.is_Integer:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Rational:
                if not quotient.is_Rational:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Float:
                if not quotient.is_Float:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
        elif self.is_NumberSymbol or self.is_Symbol or self is S.ImaginaryUnit:
            if quotient.is_Mul and len(quotient.args) == 2:
                if quotient.args[0].is_Integer and quotient.args[0].is_positive and quotient.args[1] == self:
                    return quotient
            elif quotient.is_Integer and c.is_Number:
                return quotient
        elif self.is_Add:
            cs, ps = self.primitive()
            # assert cs >= 1
            if c.is_Number and c is not S.NegativeOne:
                # assert c != 1 (handled at top)
                if cs is not S.One:
                    if c.is_negative:
                        xc = cs.extract_multiplicatively(-c)
                        if xc is not None:
                            xc = -xc
                    else:
                        xc = cs.extract_multiplicatively(c)
                    if xc is not None:
                        return xc*ps  # rely on 2-arg Mul to restore Add
                return None # |c| != 1 can only be extracted from cs
            if c == ps:
                return cs
            # check args of ps
            newargs = []
            arg: Expr
            for arg in ps.args: # type: ignore
                newarg = arg.extract_multiplicatively(c)
                if newarg is None:
                    return None # all or nothing
                newargs.append(newarg)
            if cs is not S.One:
                args = [cs*t for t in newargs]
                # args may be in different order
                return _unevaluated_Add(*args)
            else:
                return Add._from_args(newargs)
        elif self.is_Mul:
            args: list[Expr] = list(self.args) # type: ignore
            for i, arg in enumerate(args):
                newarg = arg.extract_multiplicatively(c)
                if newarg is not None:
                    args[i] = newarg
                    return Mul(*args)
        elif self.is_Pow or isinstance(self, exp):
            sb, se = self.as_base_exp()
            cb, ce = c.as_base_exp()
            if cb == sb:
                new_exp = se.extract_additively(ce)
                if new_exp is not None:
                    return Pow(sb, new_exp)
            elif c == sb:
                new_exp = se.extract_additively(1)
                if new_exp is not None:
                    return Pow(sb, new_exp)

        return None

    def extract_additively(self, c):
        """Return self - c if it's possible to subtract c from self and
        make all matching coefficients move towards zero, else return None.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> e = 2*x + 3
        >>> e.extract_additively(x + 1)
        x + 2
        >>> e.extract_additively(3*x)
        >>> e.extract_additively(4)
        >>> (y*(x + 1)).extract_additively(x + 1)
        >>> ((x + 1)*(x + 2*y + 1) + 3).extract_additively(x + 1)
        (x + 1)*(x + 2*y) + 3

        See Also
        ========
        extract_multiplicatively
        coeff
        as_coefficient

        """

        c = sympify(c)
        if self is S.NaN:
            return None
        if c.is_zero:
            return self
        elif c == self:
            return S.Zero
        elif self == S.Zero:
            return None

        if self.is_Number:
            if not c.is_Number:
                return None
            co = self
            diff = co - c
            # XXX should we match types? i.e should 3 - .1 succeed?
            if (co > 0 and diff >= 0 and diff < co or
                    co < 0 and diff <= 0 and diff > co):
                return diff
            return None

        if c.is_Number:
            co, t = self.as_coeff_Add()
            xa = co.extract_additively(c)
            if xa is None:
                return None
            return xa + t

        # handle the args[0].is_Number case separately
        # since we will have trouble looking for the coeff of
        # a number.
        if c.is_Add and c.args[0].is_Number:
            # whole term as a term factor
            co = self.coeff(c)
            xa0 = (co.extract_additively(1) or 0)*c
            if xa0:
                diff = self - co*c
                return (xa0 + (diff.extract_additively(c) or diff)) or None
            # term-wise
            h, t = c.as_coeff_Add()
            sh, st = self.as_coeff_Add()
            xa = sh.extract_additively(h)
            if xa is None:
                return None
            xa2 = st.extract_additively(t)
            if xa2 is None:
                return None
            return xa + xa2

        # whole term as a term factor
        co, diff = _corem(self, c)
        xa0 = (co.extract_additively(1) or 0)*c
        if xa0:
            return (xa0 + (diff.extract_additively(c) or diff)) or None
        # term-wise
        coeffs = []
        for a in Add.make_args(c):
            ac, at = a.as_coeff_Mul()
            co = self.coeff(at)
            if not co:
                return None
            coc, cot = co.as_coeff_Add()
            xa = coc.extract_additively(ac)
            if xa is None:
                return None
            self -= co*at
            coeffs.append((cot + xa)*at)
        coeffs.append(self)
        return Add(*coeffs)

    @property
    def expr_free_symbols(self):
        """
        Like ``free_symbols``, but returns the free symbols only if
        they are contained in an expression node.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> (x + y).expr_free_symbols # doctest: +SKIP
        {x, y}

        If the expression is contained in a non-expression object, do not return
        the free symbols. Compare:

        >>> from sympy import Tuple
        >>> t = Tuple(x + y)
        >>> t.expr_free_symbols # doctest: +SKIP
        set()
        >>> t.free_symbols
        {x, y}
        """
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        return {j for i in self.args for j in i.expr_free_symbols}

    def could_extract_minus_sign(self) -> bool:
        """Return True if self has -1 as a leading factor or has
        more literal negative signs than positive signs in a sum,
        otherwise False.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> e = x - y
        >>> {i.could_extract_minus_sign() for i in (e, -e)}
        {False, True}

        Though the ``y - x`` is considered like ``-(x - y)``, since it
        is in a product without a leading factor of -1, the result is
        false below:

        >>> (x*(y - x)).could_extract_minus_sign()
        False

        To put something in canonical form wrt to sign, use `signsimp`:

        >>> from sympy import signsimp
        >>> signsimp(x*(y - x))
        -x*(x - y)
        >>> _.could_extract_minus_sign()
        True
        """
        return False

    def extract_branch_factor(self, allow_half=False):
        """
        Try to write self as ``exp_polar(2*pi*I*n)*z`` in a nice way.
        Return (z, n).

        >>> from sympy import exp_polar, I, pi
        >>> from sympy.abc import x, y
        >>> exp_polar(I*pi).extract_branch_factor()
        (exp_polar(I*pi), 0)
        >>> exp_polar(2*I*pi).extract_branch_factor()
        (1, 1)
        >>> exp_polar(-pi*I).extract_branch_factor()
        (exp_polar(I*pi), -1)
        >>> exp_polar(3*pi*I + x).extract_branch_factor()
        (exp_polar(x + I*pi), 1)
        >>> (y*exp_polar(-5*pi*I)*exp_polar(3*pi*I + 2*pi*x)).extract_branch_factor()
        (y*exp_polar(2*pi*x), -1)
        >>> exp_polar(-I*pi/2).extract_branch_factor()
        (exp_polar(-I*pi/2), 0)

        If allow_half is True, also extract exp_polar(I*pi):

        >>> exp_polar(I*pi).extract_branch_factor(allow_half=True)
        (1, 1/2)
        >>> exp_polar(2*I*pi).extract_branch_factor(allow_half=True)
        (1, 1)
        >>> exp_polar(3*I*pi).extract_branch_factor(allow_half=True)
        (1, 3/2)
        >>> exp_polar(-I*pi).extract_branch_factor(allow_half=True)
        (1, -1/2)
        """
        from sympy.functions.elementary.exponential import exp_polar
        from sympy.functions.elementary.integers import ceiling

        n = S.Zero
        res = S.One
        args = Mul.make_args(self)
        exps = []
        for arg in args:
            if isinstance(arg, exp_polar):
                exps += [arg.exp]
            else:
                res *= arg
        piimult = S.Zero
        extras = []

        ipi = S.Pi*S.ImaginaryUnit
        while exps:
            exp = exps.pop()
            if exp.is_Add:
                exps += exp.args
                continue
            if exp.is_Mul:
                coeff = exp.as_coefficient(ipi)
                if coeff is not None:
                    piimult += coeff
                    continue
            extras += [exp]
        if piimult.is_number:
            coeff = piimult
            tail = ()
        else:
            coeff, tail = piimult.as_coeff_add(*piimult.free_symbols)
        # round down to nearest multiple of 2
        branchfact = ceiling(coeff/2 - S.Half)*2
        n += branchfact/2
        c = coeff - branchfact
        if allow_half:
            nc = c.extract_additively(1)
            if nc is not None:
                n += S.Half
                c = nc
        newexp = ipi*Add(*((c, ) + tail)) + Add(*extras)
        if newexp != 0:
            res *= exp_polar(newexp)
        return res, n

    def is_polynomial(self, *syms):
        r"""
        Return True if self is a polynomial in syms and False otherwise.

        This checks if self is an exact polynomial in syms.  This function
        returns False for expressions that are "polynomials" with symbolic
        exponents.  Thus, you should be able to apply polynomial algorithms to
        expressions for which this returns True, and Poly(expr, \*syms) should
        work if and only if expr.is_polynomial(\*syms) returns True. The
        polynomial does not have to be in expanded form.  If no symbols are
        given, all free symbols in the expression will be used.

        This is not part of the assumptions system.  You cannot do
        Symbol('z', polynomial=True).

        Examples
        ========

        >>> from sympy import Symbol, Function
        >>> x = Symbol('x')
        >>> ((x**2 + 1)**4).is_polynomial(x)
        True
        >>> ((x**2 + 1)**4).is_polynomial()
        True
        >>> (2**x + 1).is_polynomial(x)
        False
        >>> (2**x + 1).is_polynomial(2**x)
        True
        >>> f = Function('f')
        >>> (f(x) + 1).is_polynomial(x)
        False
        >>> (f(x) + 1).is_polynomial(f(x))
        True
        >>> (1/f(x) + 1).is_polynomial(f(x))
        False

        >>> n = Symbol('n', nonnegative=True, integer=True)
        >>> (x**n + 1).is_polynomial(x)
        False

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be a polynomial to
        become one.

        >>> from sympy import sqrt, factor, cancel
        >>> y = Symbol('y', positive=True)
        >>> a = sqrt(y**2 + 2*y + 1)
        >>> a.is_polynomial(y)
        False
        >>> factor(a)
        y + 1
        >>> factor(a).is_polynomial(y)
        True

        >>> b = (y**2 + 2*y + 1)/(y + 1)
        >>> b.is_polynomial(y)
        False
        >>> cancel(b)
        y + 1
        >>> cancel(b).is_polynomial(y)
        True

        See also .is_rational_function()

        """
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols
            if not syms:
                return True

        return self._eval_is_polynomial(syms)

    def _eval_is_polynomial(self, syms) -> bool | None:
        if self in syms:
            return True
        if not self.has_free(*syms):
            # constant polynomial
            return True
        # subclasses should return True or False
        return None

    def is_rational_function(self, *syms):
        """
        Test whether function is a ratio of two polynomials in the given
        symbols, syms. When syms is not given, all free symbols will be used.
        The rational function does not have to be in expanded or in any kind of
        canonical form.

        This function returns False for expressions that are "rational
        functions" with symbolic exponents.  Thus, you should be able to call
        .as_numer_denom() and apply polynomial algorithms to the result for
        expressions for which this returns True.

        This is not part of the assumptions system.  You cannot do
        Symbol('z', rational_function=True).

        Examples
        ========

        >>> from sympy import Symbol, sin
        >>> from sympy.abc import x, y

        >>> (x/y).is_rational_function()
        True

        >>> (x**2).is_rational_function()
        True

        >>> (x/sin(y)).is_rational_function(y)
        False

        >>> n = Symbol('n', integer=True)
        >>> (x**n + 1).is_rational_function(x)
        False

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be a rational function
        to become one.

        >>> from sympy import sqrt, factor
        >>> y = Symbol('y', positive=True)
        >>> a = sqrt(y**2 + 2*y + 1)/y
        >>> a.is_rational_function(y)
        False
        >>> factor(a)
        (y + 1)/y
        >>> factor(a).is_rational_function(y)
        True

        See also is_algebraic_expr().

        """
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols
            if not syms:
                return self not in _illegal

        return self._eval_is_rational_function(syms)

    def _eval_is_rational_function(self, syms) -> bool | None:
        if self in syms:
            return True
        if not self.has_xfree(syms):
            return True
        # subclasses should return True or False
        return None

    def is_meromorphic(self, x, a):
        """
        This tests whether an expression is meromorphic as
        a function of the given symbol ``x`` at the point ``a``.

        This method is intended as a quick test that will return
        None if no decision can be made without simplification or
        more detailed analysis.

        Examples
        ========

        >>> from sympy import zoo, log, sin, sqrt
        >>> from sympy.abc import x

        >>> f = 1/x**2 + 1 - 2*x**3
        >>> f.is_meromorphic(x, 0)
        True
        >>> f.is_meromorphic(x, 1)
        True
        >>> f.is_meromorphic(x, zoo)
        True

        >>> g = x**log(3)
        >>> g.is_meromorphic(x, 0)
        False
        >>> g.is_meromorphic(x, 1)
        True
        >>> g.is_meromorphic(x, zoo)
        False

        >>> h = sin(1/x)*x**2
        >>> h.is_meromorphic(x, 0)
        False
        >>> h.is_meromorphic(x, 1)
        True
        >>> h.is_meromorphic(x, zoo)
        True

        Multivalued functions are considered meromorphic when their
        branches are meromorphic. Thus most functions are meromorphic
        everywhere except at essential singularities and branch points.
        In particular, they will be meromorphic also on branch cuts
        except at their endpoints.

        >>> log(x).is_meromorphic(x, -1)
        True
        >>> log(x).is_meromorphic(x, 0)
        False
        >>> sqrt(x).is_meromorphic(x, -1)
        True
        >>> sqrt(x).is_meromorphic(x, 0)
        False

        """
        if not x.is_symbol:
            raise TypeError("{} should be of symbol type".format(x))
        a = sympify(a)

        return self._eval_is_meromorphic(x, a)

    def _eval_is_meromorphic(self, x, a) -> bool | None:
        if self == x:
            return True
        if not self.has_free(x):
            return True
        # subclasses should return True or False
        return None

    def is_algebraic_expr(self, *syms):
        """
        This tests whether a given expression is algebraic or not, in the
        given symbols, syms. When syms is not given, all free symbols
        will be used. The rational function does not have to be in expanded
        or in any kind of canonical form.

        This function returns False for expressions that are "algebraic
        expressions" with symbolic exponents. This is a simple extension to the
        is_rational_function, including rational exponentiation.

        Examples
        ========

        >>> from sympy import Symbol, sqrt
        >>> x = Symbol('x', real=True)
        >>> sqrt(1 + x).is_rational_function()
        False
        >>> sqrt(1 + x).is_algebraic_expr()
        True

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be an algebraic
        expression to become one.

        >>> from sympy import exp, factor
        >>> a = sqrt(exp(x)**2 + 2*exp(x) + 1)/(exp(x) + 1)
        >>> a.is_algebraic_expr(x)
        False
        >>> factor(a).is_algebraic_expr()
        True

        See Also
        ========

        is_rational_function

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Algebraic_expression

        """
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols
            if not syms:
                return True

        return self._eval_is_algebraic_expr(syms)

    def _eval_is_algebraic_expr(self, syms) -> bool | None:
        if self in syms:
            return True
        if not self.has_free(*syms):
            return True
        # subclasses should return True or False
        return None

    ###################################################################################
    ##################### SERIES, LEADING TERM, LIMIT, ORDER METHODS ##################
    ###################################################################################

    def series(self, x=None, x0=0, n=6, dir="+", logx=None, cdir=0):
        """
        Series expansion of "self" around ``x = x0`` yielding either terms of
        the series one by one (the lazy series given when n=None), else
        all the terms at once when n != None.

        Returns the series expansion of "self" around the point ``x = x0``
        with respect to ``x`` up to ``O((x - x0)**n, x, x0)`` (default n is 6).

        If ``x=None`` and ``self`` is univariate, the univariate symbol will
        be supplied, otherwise an error will be raised.

        Parameters
        ==========

        expr : Expression
               The expression whose series is to be expanded.

        x : Symbol
            It is the variable of the expression to be calculated.

        x0 : Value
             The value around which ``x`` is calculated. Can be any value
             from ``-oo`` to ``oo``.

        n : Value
            The value used to represent the order in terms of ``x**n``,
            up to which the series is to be expanded.

        dir : String, optional
              The series-expansion can be bi-directional. If ``dir="+"``,
              then (x->x0+). If ``dir="-", then (x->x0-). For infinite
              ``x0`` (``oo`` or ``-oo``), the ``dir`` argument is determined
              from the direction of the infinity (i.e., ``dir="-"`` for
              ``oo``).

        logx : optional
               It is used to replace any log(x) in the returned series with a
               symbolic value rather than evaluating the actual value.

        cdir : optional
               It stands for complex direction, and indicates the direction
               from which the expansion needs to be evaluated.

        Examples
        ========

        >>> from sympy import cos, exp, tan
        >>> from sympy.abc import x, y
        >>> cos(x).series()
        1 - x**2/2 + x**4/24 + O(x**6)
        >>> cos(x).series(n=4)
        1 - x**2/2 + O(x**4)
        >>> cos(x).series(x, x0=1, n=2)
        cos(1) - (x - 1)*sin(1) + O((x - 1)**2, (x, 1))
        >>> e = cos(x + exp(y))
        >>> e.series(y, n=2)
        cos(x + 1) - y*sin(x + 1) + O(y**2)
        >>> e.series(x, n=2)
        cos(exp(y)) - x*sin(exp(y)) + O(x**2)

        If ``n=None`` then a generator of the series terms will be returned.

        >>> term=cos(x).series(n=None)
        >>> [next(term) for i in range(2)]
        [1, -x**2/2]

        For ``dir=+`` (default) the series is calculated from the right and
        for ``dir=-`` the series from the left. For smooth functions this
        flag will not alter the results.

        >>> abs(x).series(dir="+")
        x
        >>> abs(x).series(dir="-")
        -x
        >>> f = tan(x)
        >>> f.series(x, 2, 6, "+")
        tan(2) + (1 + tan(2)**2)*(x - 2) + (x - 2)**2*(tan(2)**3 + tan(2)) +
        (x - 2)**3*(1/3 + 4*tan(2)**2/3 + tan(2)**4) + (x - 2)**4*(tan(2)**5 +
        5*tan(2)**3/3 + 2*tan(2)/3) + (x - 2)**5*(2/15 + 17*tan(2)**2/15 +
        2*tan(2)**4 + tan(2)**6) + O((x - 2)**6, (x, 2))

        >>> f.series(x, 2, 3, "-")
        tan(2) + (2 - x)*(-tan(2)**2 - 1) + (2 - x)**2*(tan(2)**3 + tan(2))
        + O((x - 2)**3, (x, 2))

        For rational expressions this method may return original expression without the Order term.
        >>> (1/x).series(x, n=8)
        1/x

        Returns
        =======

        Expr : Expression
            Series expansion of the expression about x0

        Raises
        ======

        TypeError
            If "n" and "x0" are infinity objects

        PoleError
            If "x0" is an infinity object

        """
        if x is None:
            syms = self.free_symbols
            if not syms:
                return self
            elif len(syms) > 1:
                raise ValueError('x must be given for multivariate functions.')
            x = syms.pop()

        from .symbol import Dummy, Symbol
        if isinstance(x, Symbol):
            dep = x in self.free_symbols
        else:
            d = Dummy()
            dep = d in self.xreplace({x: d}).free_symbols
        if not dep:
            if n is None:
                return (s for s in [self])
            else:
                return self

        if len(dir) != 1 or dir not in '+-':
            raise ValueError("Dir must be '+' or '-'")

        if n is not None:
            n = int(n)
            if n < 0:
                raise ValueError("Number of terms should be nonnegative")

        x0 = sympify(x0)
        cdir = sympify(cdir)
        from sympy.functions.elementary.complexes import im, sign

        if not cdir.is_zero:
            if cdir.is_real:
                dir = '+' if cdir.is_positive else '-'
            else:
                dir = '+' if im(cdir).is_positive else '-'
        else:
            if x0 and x0.is_infinite:
                cdir = sign(x0).simplify()
            elif str(dir) == "+":
                cdir = S.One
            elif str(dir) == "-":
                cdir = S.NegativeOne
            elif cdir == S.Zero:
                cdir = S.One

        cdir = cdir/abs(cdir)

        if x0 and x0.is_infinite:
            from .function import PoleError
            try:
                s = self.subs(x, cdir/x).series(x, n=n, dir='+', cdir=1)
                if n is None:
                    return (si.subs(x, cdir/x) for si in s)
                return s.subs(x, cdir/x)
            except PoleError:
                s = self.subs(x, cdir*x).aseries(x, n=n)
                return s.subs(x, cdir*x)

        # use rep to shift origin to x0 and change sign (if dir is negative)
        # and undo the process with rep2
        if x0 or cdir != 1:
            s = self.subs({x: x0 + cdir*x}).series(x, x0=0, n=n, dir='+', logx=logx, cdir=1)
            if n is None:  # lseries...
                return (si.subs({x: x/cdir - x0/cdir}) for si in s)
            return s.subs({x: x/cdir - x0/cdir})

        # from here on it's x0=0 and dir='+' handling

        if x.is_positive is x.is_negative is None or x.is_Symbol is not True:
            # replace x with an x that has a positive assumption
            xpos = Dummy('x', positive=True)
            rv = self.subs(x, xpos).series(xpos, x0, n, dir, logx=logx, cdir=cdir)
            if n is None:
                return (s.subs(xpos, x) for s in rv)
            else:
                return rv.subs(xpos, x)

        from sympy.series.order import Order
        if n is not None:  # nseries handling
            s1 = self._eval_nseries(x, n=n, logx=logx, cdir=cdir)
            o = s1.getO() or S.Zero
            if o:
                # make sure the requested order is returned
                ngot = o.getn()
                if ngot > n:
                    # leave o in its current form (e.g. with x*log(x)) so
                    # it eats terms properly, then replace it below
                    if n != 0:
                        s1 += o.subs(x, x**Rational(n, ngot))
                    else:
                        s1 += Order(1, x)
                elif ngot < n:
                    # increase the requested number of terms to get the desired
                    # number keep increasing (up to 9) until the received order
                    # is different than the original order and then predict how
                    # many additional terms are needed
                    from sympy.functions.elementary.integers import ceiling
                    for more in range(1, 9):
                        s1 = self._eval_nseries(x, n=n + more, logx=logx, cdir=cdir)
                        newn = s1.getn()
                        if newn != ngot:
                            ndo = n + ceiling((n - ngot)*more/(newn - ngot))
                            s1 = self._eval_nseries(x, n=ndo, logx=logx, cdir=cdir)
                            while s1.getn() < n:
                                s1 = self._eval_nseries(x, n=ndo, logx=logx, cdir=cdir)
                                ndo += 1
                            break
                    else:
                        raise ValueError('Could not calculate %s terms for %s'
                                         % (str(n), self))
                    s1 += Order(x**n, x)
                o = s1.getO()
                s1 = s1.removeO()
            elif s1.has(Order):
                # asymptotic expansion
                return s1
            else:
                o = Order(x**n, x)
                s1done = s1.doit()
                try:
                    if (s1done + o).removeO() == s1done:
                        o = S.Zero
                except NotImplementedError:
                    return s1

            try:
                from sympy.simplify.radsimp import collect
                return collect(s1, x) + o
            except NotImplementedError:
                return s1 + o

        else:  # lseries handling
            def yield_lseries(s):
                """Return terms of lseries one at a time."""
                for si in s:
                    if not si.is_Add:
                        yield si
                        continue
                    # yield terms 1 at a time if possible
                    # by increasing order until all the
                    # terms have been returned
                    yielded = 0
                    o = Order(si, x)*x
                    ndid = 0
                    ndo = len(si.args)
                    while 1:
                        do = (si - yielded + o).removeO()
                        o *= x
                        if not do or do.is_Order:
                            continue
                        if do.is_Add:
                            ndid += len(do.args)
                        else:
                            ndid += 1
                        yield do
                        if ndid == ndo:
                            break
                        yielded += do

            return yield_lseries(self.removeO()._eval_lseries(x, logx=logx, cdir=cdir))

    def aseries(self, x=None, n=6, bound=0, hir=False):
        """Asymptotic Series expansion of self.
        This is equivalent to ``self.series(x, oo, n)``.

        Parameters
        ==========

        self : Expression
               The expression whose series is to be expanded.

        x : Symbol
            It is the variable of the expression to be calculated.

        n : Value
            The value used to represent the order in terms of ``x**n``,
            up to which the series is to be expanded.

        hir : Boolean
              Set this parameter to be True to produce hierarchical series.
              It stops the recursion at an early level and may provide nicer
              and more useful results.

        bound : Value, Integer
                Use the ``bound`` parameter to give limit on rewriting
                coefficients in its normalised form.

        Examples
        ========

        >>> from sympy import sin, exp
        >>> from sympy.abc import x

        >>> e = sin(1/x + exp(-x)) - sin(1/x)

        >>> e.aseries(x)
        (1/(24*x**4) - 1/(2*x**2) + 1 + O(x**(-6), (x, oo)))*exp(-x)

        >>> e.aseries(x, n=3, hir=True)
        -exp(-2*x)*sin(1/x)/2 + exp(-x)*cos(1/x) + O(exp(-3*x), (x, oo))

        >>> e = exp(exp(x)/(1 - 1/x))

        >>> e.aseries(x)
        exp(exp(x)/(1 - 1/x))

        >>> e.aseries(x, bound=3) # doctest: +SKIP
        exp(exp(x)/x**2)*exp(exp(x)/x)*exp(-exp(x) + exp(x)/(1 - 1/x) - exp(x)/x - exp(x)/x**2)*exp(exp(x))

        For rational expressions this method may return original expression without the Order term.
        >>> (1/x).aseries(x, n=8)
        1/x

        Returns
        =======

        Expr
            Asymptotic series expansion of the expression.

        Notes
        =====

        This algorithm is directly induced from the limit computational algorithm provided by Gruntz.
        It majorly uses the mrv and rewrite sub-routines. The overall idea of this algorithm is first
        to look for the most rapidly varying subexpression w of a given expression f and then expands f
        in a series in w. Then same thing is recursively done on the leading coefficient
        till we get constant coefficients.

        If the most rapidly varying subexpression of a given expression f is f itself,
        the algorithm tries to find a normalised representation of the mrv set and rewrites f
        using this normalised representation.

        If the expansion contains an order term, it will be either ``O(x ** (-n))`` or ``O(w ** (-n))``
        where ``w`` belongs to the most rapidly varying expression of ``self``.

        References
        ==========

        .. [1] Gruntz, Dominik. A new algorithm for computing asymptotic series.
               In: Proc. 1993 Int. Symp. Symbolic and Algebraic Computation. 1993.
               pp. 239-244.
        .. [2] Gruntz thesis - p90
        .. [3] https://en.wikipedia.org/wiki/Asymptotic_expansion

        See Also
        ========

        Expr.aseries: See the docstring of this function for complete details of this wrapper.
        """

        from .symbol import Dummy

        if x.is_positive is x.is_negative is None:
            xpos = Dummy('x', positive=True)
            return self.subs(x, xpos).aseries(xpos, n, bound, hir).subs(xpos, x)

        from .function import PoleError
        from sympy.series.gruntz import mrv, rewrite

        try:
            om, exps = mrv(self, x)
        except PoleError:
            return self

        # We move one level up by replacing `x` by `exp(x)`, and then
        # computing the asymptotic series for f(exp(x)). Then asymptotic series
        # can be obtained by moving one-step back, by replacing x by ln(x).

        from sympy.functions.elementary.exponential import exp, log
        from sympy.series.order import Order

        if x in om:
            s = self.subs(x, exp(x)).aseries(x, n, bound, hir).subs(x, log(x))
            if s.getO():
                return s + Order(1/x**n, (x, S.Infinity))
            return s

        k = Dummy('k', positive=True)
        # f is rewritten in terms of omega
        func, logw = rewrite(exps, om, x, k)

        if self in om:
            if bound <= 0:
                return self
            s = (self.exp).aseries(x, n, bound=bound)
            s = s.func(*[t.removeO() for t in s.args])
            try:
                res = exp(s.subs(x, 1/x).as_leading_term(x).subs(x, 1/x))
            except PoleError:
                res = self

            func = exp(self.args[0] - res.args[0]) / k
            logw = log(1/res)

        s = func.series(k, 0, n)
        from sympy.core.function import expand_mul
        s = expand_mul(s)
        # Hierarchical series
        if hir:
            return s.subs(k, exp(logw))

        o = s.getO()
        terms = sorted(Add.make_args(s.removeO()), key=lambda i: int(i.as_coeff_exponent(k)[1]))
        s = S.Zero
        has_ord = False

        # Then we recursively expand these coefficients one by one into
        # their asymptotic series in terms of their most rapidly varying subexpressions.
        for t in terms:
            coeff, expo = t.as_coeff_exponent(k)
            if coeff.has(x):
                # Recursive step
                snew = coeff.aseries(x, n, bound=bound-1)
                if has_ord and snew.getO():
                    break
                elif snew.getO():
                    has_ord = True
                s += (snew * k**expo)
            else:
                s += t

        if not o or has_ord:
            return s.subs(k, exp(logw))
        return (s + o).subs(k, exp(logw))


    def taylor_term(self, n, x, *previous_terms):
        """General method for the taylor term.

        This method is slow, because it differentiates n-times. Subclasses can
        redefine it to make it faster by using the "previous_terms".
        """
        from .symbol import Dummy
        from sympy.functions.combinatorial.factorials import factorial

        x = sympify(x)
        _x = Dummy('x')
        return self.subs(x, _x).diff(_x, n).subs(_x, x).subs(x, 0) * x**n / factorial(n)

    def lseries(self, x=None, x0=0, dir='+', logx=None, cdir=0):
        """
        Wrapper for series yielding an iterator of the terms of the series.

        Note: an infinite series will yield an infinite iterator. The following,
        for exaxmple, will never terminate. It will just keep printing terms
        of the sin(x) series::

          for term in sin(x).lseries(x):
              print term

        The advantage of lseries() over nseries() is that many times you are
        just interested in the next term in the series (i.e. the first term for
        example), but you do not know how many you should ask for in nseries()
        using the "n" parameter.

        See also nseries().
        """
        return self.series(x, x0, n=None, dir=dir, logx=logx, cdir=cdir)

    def _eval_lseries(self, x, logx=None, cdir=0):
        # default implementation of lseries is using nseries(), and adaptively
        # increasing the "n". As you can see, it is not very efficient, because
        # we are calculating the series over and over again. Subclasses should
        # override this method and implement much more efficient yielding of
        # terms.
        n = 0
        series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        while series.is_Order:
            n += 1
            series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        e = series.removeO()
        yield e
        if e is S.Zero:
            return

        while 1:
            while 1:
                n += 1
                series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir).removeO()
                if e != series:
                    break
                if (series - self).cancel() is S.Zero:
                    return
            yield series - e
            e = series

    def nseries(self, x=None, x0=0, n=6, dir='+', logx=None, cdir=0):
        """
        Wrapper to _eval_nseries if assumptions allow, else to series.

        If x is given, x0 is 0, dir='+', and self has x, then _eval_nseries is
        called. This calculates "n" terms in the innermost expressions and
        then builds up the final series just by "cross-multiplying" everything
        out.

        The optional ``logx`` parameter can be used to replace any log(x) in the
        returned series with a symbolic value to avoid evaluating log(x) at 0. A
        symbol to use in place of log(x) should be provided.

        Advantage -- it's fast, because we do not have to determine how many
        terms we need to calculate in advance.

        Disadvantage -- you may end up with less terms than you may have
        expected, but the O(x**n) term appended will always be correct and
        so the result, though perhaps shorter, will also be correct.

        If any of those assumptions is not met, this is treated like a
        wrapper to series which will try harder to return the correct
        number of terms.

        See also lseries().

        Examples
        ========

        >>> from sympy import sin, log, Symbol
        >>> from sympy.abc import x, y
        >>> sin(x).nseries(x, 0, 6)
        x - x**3/6 + x**5/120 + O(x**6)
        >>> log(x+1).nseries(x, 0, 5)
        x - x**2/2 + x**3/3 - x**4/4 + O(x**5)

        Handling of the ``logx`` parameter --- in the following example the
        expansion fails since ``sin`` does not have an asymptotic expansion
        at -oo (the limit of log(x) as x approaches 0):

        >>> e = sin(log(x))
        >>> e.nseries(x, 0, 6)
        Traceback (most recent call last):
        ...
        PoleError: ...
        ...
        >>> logx = Symbol('logx')
        >>> e.nseries(x, 0, 6, logx=logx)
        sin(logx)

        In the following example, the expansion works but only returns self
        unless the ``logx`` parameter is used:

        >>> e = x**y
        >>> e.nseries(x, 0, 2)
        x**y
        >>> e.nseries(x, 0, 2, logx=logx)
        exp(logx*y)

        """
        if x and x not in self.free_symbols:
            return self
        if x is None or x0 or dir != '+':  # {see XPOS above} or (x.is_positive == x.is_negative == None):
            return self.series(x, x0, n, dir, cdir=cdir)
        else:
            return self._eval_nseries(x, n=n, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir):
        """
        Return terms of series for self up to O(x**n) at x=0
        from the positive direction.

        This is a method that should be overridden in subclasses. Users should
        never call this method directly (use .nseries() instead), so you do not
        have to write docstrings for _eval_nseries().
        """
        raise NotImplementedError(filldedent("""
                     The _eval_nseries method should be added to
                     %s to give terms up to O(x**n) at x=0
                     from the positive direction so it is available when
                     nseries calls it.""" % self.func)
                     )

    def limit(self, x, xlim, dir='+'):
        """ Compute limit x->xlim.
        """
        from sympy.series.limits import limit
        return limit(self, x, xlim, dir)

    def compute_leading_term(self, x, logx=None):
        """Deprecated function to compute the leading term of a series.

        as_leading_term is only allowed for results of .series()
        This is a wrapper to compute a series first.
        """
        from sympy.utilities.exceptions import SymPyDeprecationWarning

        SymPyDeprecationWarning(
            feature="compute_leading_term",
            useinstead="as_leading_term",
            issue=21843,
            deprecated_since_version="1.12"
        ).warn()

        from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
        if self.has(Piecewise):
            expr = piecewise_fold(self)
        else:
            expr = self
        if self.removeO() == 0:
            return self

        from .symbol import Dummy
        from sympy.functions.elementary.exponential import log
        from sympy.series.order import Order

        _logx = logx
        logx = Dummy('logx') if logx is None else logx
        res = Order(1)
        incr = S.One
        while res.is_Order:
            res = expr._eval_nseries(x, n=1+incr, logx=logx).cancel().powsimp().trigsimp()
            incr *= 2

        if _logx is None:
            res = res.subs(logx, log(x))

        return res.as_leading_term(x)

    @cacheit
    def as_leading_term(self, *symbols, logx=None, cdir=0):
        """
        Returns the leading (nonzero) term of the series expansion of self.

        The _eval_as_leading_term routines are used to do this, and they must
        always return a non-zero value.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (1 + x + x**2).as_leading_term(x)
        1
        >>> (1/x**2 + x + x**2).as_leading_term(x)
        x**(-2)

        """
        if len(symbols) > 1:
            c = self
            for x in symbols:
                c = c.as_leading_term(x, logx=logx, cdir=cdir)
            return c
        elif not symbols:
            return self
        x = sympify(symbols[0])
        cdir = sympify(cdir)
        if not x.is_symbol:
            raise ValueError('expecting a Symbol but got %s' % x)
        if x not in self.free_symbols:
            return self
        obj = self._eval_as_leading_term(x, logx=logx, cdir=cdir)
        if obj is not None:
            from sympy.simplify.powsimp import powsimp
            return powsimp(obj, deep=True, combine='exp')
        raise NotImplementedError('as_leading_term(%s, %s)' % (self, x))

    def _eval_as_leading_term(self, x, logx, cdir):
        return self

    def as_coeff_exponent(self, x) -> tuple[Expr, Expr]:
        """ ``c*x**e -> c,e`` where x can be any symbolic expression.
        """
        from sympy.simplify.radsimp import collect
        s = collect(self, x)
        c, p = s.as_coeff_mul(x)
        if len(p) == 1:
            b, e = p[0].as_base_exp()
            if b == x:
                return c, e
        return s, S.Zero

    def leadterm(self, x, logx=None, cdir=0):
        """
        Returns the leading term a*x**b as a tuple (a, b).

        Examples
        ========

        >>> from sympy.abc import x
        >>> (1+x+x**2).leadterm(x)
        (1, 0)
        >>> (1/x**2+x+x**2).leadterm(x)
        (1, -2)

        """
        from .symbol import Dummy
        from sympy.functions.elementary.exponential import log
        l = self.as_leading_term(x, logx=logx, cdir=cdir)
        d = Dummy('logx')
        if l.has(log(x)):
            l = l.subs(log(x), d)
        c, e = l.as_coeff_exponent(x)
        if x in c.free_symbols:
            raise ValueError(filldedent("""
                cannot compute leadterm(%s, %s). The coefficient
                should have been free of %s but got %s""" % (self, x, x, c)))
        c = c.subs(d, log(x))
        return c, e

    def as_coeff_Mul(self, rational: bool = False) -> tuple['Number', Expr]:
        """Efficiently extract the coefficient of a product."""
        return S.One, self

    def as_coeff_Add(self, rational=False) -> tuple['Number', Expr]:
        """Efficiently extract the coefficient of a summation."""
        return S.Zero, self

    def fps(self, x=None, x0=0, dir=1, hyper=True, order=4, rational=True,
            full=False):
        """
        Compute formal power power series of self.

        See the docstring of the :func:`fps` function in sympy.series.formal for
        more information.
        """
        from sympy.series.formal import fps

        return fps(self, x, x0, dir, hyper, order, rational, full)

    def fourier_series(self, limits=None):
        """Compute fourier sine/cosine series of self.

        See the docstring of the :func:`fourier_series` in sympy.series.fourier
        for more information.
        """
        from sympy.series.fourier import fourier_series

        return fourier_series(self, limits)

    ###################################################################################
    ##################### DERIVATIVE, INTEGRAL, FUNCTIONAL METHODS ####################
    ###################################################################################

    def diff(self, *symbols, **assumptions):
        assumptions.setdefault("evaluate", True)
        return _derivative_dispatch(self, *symbols, **assumptions)

    ###########################################################################
    ###################### EXPRESSION EXPANSION METHODS #######################
    ###########################################################################

    # Relevant subclasses should override _eval_expand_hint() methods.  See
    # the docstring of expand() for more info.

    def _eval_expand_complex(self, **hints):
        real, imag = self.as_real_imag(**hints)
        return real + S.ImaginaryUnit*imag

    @staticmethod
    def _expand_hint(expr, hint, deep=True, **hints):
        """
        Helper for ``expand()``.  Recursively calls ``expr._eval_expand_hint()``.

        Returns ``(expr, hit)``, where expr is the (possibly) expanded
        ``expr`` and ``hit`` is ``True`` if ``expr`` was truly expanded and
        ``False`` otherwise.
        """
        hit = False
        # XXX: Hack to support non-Basic args
        #              |
        #              V
        if deep and getattr(expr, 'args', ()) and not expr.is_Atom:
            sargs = []
            for arg in expr.args:
                arg, arghit = Expr._expand_hint(arg, hint, **hints)
                hit |= arghit
                sargs.append(arg)

            if hit:
                expr = expr.func(*sargs)

        if hasattr(expr, hint):
            newexpr = getattr(expr, hint)(**hints)
            if newexpr != expr:
                return (newexpr, True)

        return (expr, hit)

    @cacheit
    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True,
            mul=True, log=True, multinomial=True, basic=True, **hints):
        """
        Expand an expression using hints.

        See the docstring of the expand() function in sympy.core.function for
        more information.

        """
        from sympy.simplify.radsimp import fraction

        hints.update(power_base=power_base, power_exp=power_exp, mul=mul,
           log=log, multinomial=multinomial, basic=basic)

        expr = self
        # default matches fraction's default
        _fraction = lambda x: fraction(x, hints.get('exact', False))
        if hints.pop('frac', False):
            n, d = [a.expand(deep=deep, modulus=modulus, **hints)
                    for a in _fraction(self)]
            return n/d
        elif hints.pop('denom', False):
            n, d = _fraction(self)
            return n/d.expand(deep=deep, modulus=modulus, **hints)
        elif hints.pop('numer', False):
            n, d = _fraction(self)
            return n.expand(deep=deep, modulus=modulus, **hints)/d

        # Although the hints are sorted here, an earlier hint may get applied
        # at a given node in the expression tree before another because of how
        # the hints are applied.  e.g. expand(log(x*(y + z))) -> log(x*y +
        # x*z) because while applying log at the top level, log and mul are
        # applied at the deeper level in the tree so that when the log at the
        # upper level gets applied, the mul has already been applied at the
        # lower level.

        # Additionally, because hints are only applied once, the expression
        # may not be expanded all the way.   For example, if mul is applied
        # before multinomial, x*(x + 1)**2 won't be expanded all the way.  For
        # now, we just use a special case to make multinomial run before mul,
        # so that at least polynomials will be expanded all the way.  In the
        # future, smarter heuristics should be applied.
        # TODO: Smarter heuristics

        def _expand_hint_key(hint):
            """Make multinomial come before mul"""
            if hint == 'mul':
                return 'mulz'
            return hint

        for hint in sorted(hints.keys(), key=_expand_hint_key):
            use_hint = hints[hint]
            if use_hint:
                hint = '_eval_expand_' + hint
                expr, hit = Expr._expand_hint(expr, hint, deep=deep, **hints)

        while True:
            was = expr
            if hints.get('multinomial', False):
                expr, _ = Expr._expand_hint(
                    expr, '_eval_expand_multinomial', deep=deep, **hints)
            if hints.get('mul', False):
                expr, _ = Expr._expand_hint(
                    expr, '_eval_expand_mul', deep=deep, **hints)
            if hints.get('log', False):
                expr, _ = Expr._expand_hint(
                    expr, '_eval_expand_log', deep=deep, **hints)
            if expr == was:
                break

        if modulus is not None:
            modulus = sympify(modulus)

            if not modulus.is_Integer or modulus <= 0:
                raise ValueError(
                    "modulus must be a positive integer, got %s" % modulus)

            terms = []

            for term in Add.make_args(expr):
                coeff, tail = term.as_coeff_Mul(rational=True)

                coeff %= modulus

                if coeff:
                    terms.append(coeff*tail)

            expr = Add(*terms)

        return expr

    ###########################################################################
    ################### GLOBAL ACTION VERB WRAPPER METHODS ####################
    ###########################################################################

    def integrate(self, *args, **kwargs):
        """See the integrate function in sympy.integrals"""
        from sympy.integrals.integrals import integrate
        return integrate(self, *args, **kwargs)

    def nsimplify(self, constants=(), tolerance=None, full=False):
        """See the nsimplify function in sympy.simplify"""
        from sympy.simplify.simplify import nsimplify
        return nsimplify(self, constants, tolerance, full)

    def separate(self, deep=False, force=False):
        """See the separate function in sympy.simplify"""
        from .function import expand_power_base
        return expand_power_base(self, deep=deep, force=force)

    def collect(self, syms, func=None, evaluate=True, exact=False, distribute_order_term=True):
        """See the collect function in sympy.simplify"""
        from sympy.simplify.radsimp import collect
        return collect(self, syms, func, evaluate, exact, distribute_order_term)

    def together(self, *args, **kwargs):
        """See the together function in sympy.polys"""
        from sympy.polys.rationaltools import together
        return together(self, *args, **kwargs)

    def apart(self, x=None, **args):
        """See the apart function in sympy.polys"""
        from sympy.polys.partfrac import apart
        return apart(self, x, **args)

    def ratsimp(self):
        """See the ratsimp function in sympy.simplify"""
        from sympy.simplify.ratsimp import ratsimp
        return ratsimp(self)

    def trigsimp(self, **args):
        """See the trigsimp function in sympy.simplify"""
        from sympy.simplify.trigsimp import trigsimp
        return trigsimp(self, **args)

    def radsimp(self, **kwargs):
        """See the radsimp function in sympy.simplify"""
        from sympy.simplify.radsimp import radsimp
        return radsimp(self, **kwargs)

    def powsimp(self, *args, **kwargs):
        """See the powsimp function in sympy.simplify"""
        from sympy.simplify.powsimp import powsimp
        return powsimp(self, *args, **kwargs)

    def combsimp(self):
        """See the combsimp function in sympy.simplify"""
        from sympy.simplify.combsimp import combsimp
        return combsimp(self)

    def gammasimp(self):
        """See the gammasimp function in sympy.simplify"""
        from sympy.simplify.gammasimp import gammasimp
        return gammasimp(self)

    def factor(self, *gens, **args):
        """See the factor() function in sympy.polys.polytools"""
        from sympy.polys.polytools import factor
        return factor(self, *gens, **args)

    def cancel(self, *gens, **args):
        """See the cancel function in sympy.polys"""
        from sympy.polys.polytools import cancel
        return cancel(self, *gens, **args)

    def invert(self, g, *gens, **args):
        """Return the multiplicative inverse of ``self`` mod ``g``
        where ``self`` (and ``g``) may be symbolic expressions).

        See Also
        ========
        sympy.core.intfunc.mod_inverse, sympy.polys.polytools.invert
        """
        if self.is_number and getattr(g, 'is_number', True):
            return mod_inverse(self, g)
        from sympy.polys.polytools import invert
        return invert(self, g, *gens, **args)

    def round(self, n=None):
        """Return x rounded to the given decimal place.

        If a complex number would results, apply round to the real
        and imaginary components of the number.

        Examples
        ========

        >>> from sympy import pi, E, I, S, Number
        >>> pi.round()
        3
        >>> pi.round(2)
        3.14
        >>> (2*pi + E*I).round()
        6 + 3*I

        The round method has a chopping effect:

        >>> (2*pi + I/10).round()
        6
        >>> (pi/10 + 2*I).round()
        2*I
        >>> (pi/10 + E*I).round(2)
        0.31 + 2.72*I

        Notes
        =====

        The Python ``round`` function uses the SymPy ``round`` method so it
        will always return a SymPy number (not a Python float or int):

        >>> isinstance(round(S(123), -2), Number)
        True
        """
        x = self

        if not x.is_number:
            raise TypeError("Cannot round symbolic expression")
        if not x.is_Atom:
            if not pure_complex(x.n(2), or_real=True):
                raise TypeError(
                    'Expected a number but got %s:' % func_name(x))
        elif x in _illegal:
            return x
        if not (xr := x.is_extended_real):
            r, i = x.as_real_imag()
            if xr is False:
                return r.round(n) + S.ImaginaryUnit*i.round(n)
            if i.equals(0):
                return r.round(n)
        if not x:
            return S.Zero if n is None else x

        p = as_int(n or 0)

        if x.is_Integer:
            return Integer(round(int(x), p))

        digits_to_decimal = _mag(x)  # _mag(12) = 2, _mag(.012) = -1
        allow = digits_to_decimal + p
        precs = [f._prec for f in x.atoms(Float)]
        dps = prec_to_dps(max(precs)) if precs else None
        if dps is None:
            # assume everything is exact so use the Python
            # float default or whatever was requested
            dps = max(15, allow)
        else:
            allow = min(allow, dps)
        # this will shift all digits to right of decimal
        # and give us dps to work with as an int
        shift = -digits_to_decimal + dps
        extra = 1  # how far we look past known digits
        # NOTE
        # mpmath will calculate the binary representation to
        # an arbitrary number of digits but we must base our
        # answer on a finite number of those digits, e.g.
        # .575 2589569785738035/2**52 in binary.
        # mpmath shows us that the first 18 digits are
        #     >>> Float(.575).n(18)
        #     0.574999999999999956
        # The default precision is 15 digits and if we ask
        # for 15 we get
        #     >>> Float(.575).n(15)
        #     0.575000000000000
        # mpmath handles rounding at the 15th digit. But we
        # need to be careful since the user might be asking
        # for rounding at the last digit and our semantics
        # are to round toward the even final digit when there
        # is a tie. So the extra digit will be used to make
        # that decision. In this case, the value is the same
        # to 15 digits:
        #     >>> Float(.575).n(16)
        #     0.5750000000000000
        # Now converting this to the 15 known digits gives
        #     575000000000000.0
        # which rounds to integer
        #    5750000000000000
        # And now we can round to the desired digt, e.g. at
        # the second from the left and we get
        #    5800000000000000
        # and rescaling that gives
        #    0.58
        # as the final result.
        # If the value is made slightly less than 0.575 we might
        # still obtain the same value:
        #    >>> Float(.575-1e-16).n(16)*10**15
        #    574999999999999.8
        # What 15 digits best represents the known digits (which are
        # to the left of the decimal? 5750000000000000, the same as
        # before. The only way we will round down (in this case) is
        # if we declared that we had more than 15 digits of precision.
        # For example, if we use 16 digits of precision, the integer
        # we deal with is
        #    >>> Float(.575-1e-16).n(17)*10**16
        #    5749999999999998.4
        # and this now rounds to 5749999999999998 and (if we round to
        # the 2nd digit from the left) we get 5700000000000000.
        #
        xf = x.n(dps + extra)*Pow(10, shift)
        if xf.is_Number and xf._prec == 1:  # xf.is_Add will raise below
            # is x == 0?
            if x.equals(0):
                return Float(0)
            raise ValueError('not computing with precision')
        xi = Integer(xf)
        # use the last digit to select the value of xi
        # nearest to x before rounding at the desired digit
        sign = 1 if x > 0 else -1
        dif2 = sign*(xf - xi).n(extra)
        if dif2 < 0:
            raise NotImplementedError(
                'not expecting int(x) to round away from 0')
        if dif2 > .5:
            xi += sign  # round away from 0
        elif dif2 == .5:
            xi += sign if xi%2 else -sign  # round toward even
        # shift p to the new position
        ip = p - shift
        # let Python handle the int rounding then rescale
        xr = round(xi.p, ip)
        # restore scale
        rv = Rational(xr, Pow(10, shift))
        # return Float or Integer
        if rv.is_Integer:
            if n is None:  # the single-arg case
                return rv
            # use str or else it won't be a float
            return Float(str(rv), dps)  # keep same precision
        else:
            if not allow and rv > self:
                allow += 1
            return Float(rv, allow)

    __round__ = round

    def _eval_derivative_matrix_lines(self, x):
        from sympy.matrices.expressions.matexpr import _LeftRightArgs
        return [_LeftRightArgs([S.One, S.One], higher=self._eval_derivative(x))]


class AtomicExpr(Atom, Expr):
    """
    A parent class for object which are both atoms and Exprs.

    For example: Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """
    is_number = False
    is_Atom = True

    __slots__ = ()

    def _eval_derivative(self, s):
        if self == s:
            return S.One
        return S.Zero

    def _eval_derivative_n_times(self, s, n):
        from .containers import Tuple
        from sympy.matrices.expressions.matexpr import MatrixExpr
        from sympy.matrices.matrixbase import MatrixBase
        if isinstance(s, (MatrixBase, Tuple, Iterable, MatrixExpr)):
            return super()._eval_derivative_n_times(s, n)
        from .relational import Eq
        from sympy.functions.elementary.piecewise import Piecewise
        if self == s:
            return Piecewise((self, Eq(n, 0)), (1, Eq(n, 1)), (0, True))
        else:
            return Piecewise((self, Eq(n, 0)), (0, True))

    def _eval_is_polynomial(self, syms):
        return True

    def _eval_is_rational_function(self, syms):
        return self not in _illegal

    def _eval_is_meromorphic(self, x, a):
        from sympy.calculus.accumulationbounds import AccumBounds
        return (not self.is_Number or self.is_finite) and not isinstance(self, AccumBounds)

    def _eval_is_algebraic_expr(self, syms):
        return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        return self

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        return {self}





@sympify_method_args
class Boolean(Basic):
    """A Boolean object is an object for which logic operations make sense."""

    __slots__ = ()

    kind = BooleanKind

    if TYPE_CHECKING:

        def __new__(cls, *args: Basic | complex) -> Boolean:
            ...

        @overload # type: ignore
        def subs(self, arg1: Mapping[Basic | complex, Boolean | complex], arg2: None=None) -> Boolean: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Boolean | complex]], arg2: None=None, **kwargs: Any) -> Boolean: ...
        @overload
        def subs(self, arg1: Boolean | complex, arg2: Boolean | complex) -> Boolean: ...
        @overload
        def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

        def subs(self, arg1: Mapping[Basic | complex, Basic | complex] | Basic | complex, # type: ignore
                 arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
            ...

        def simplify(self, **kwargs) -> Boolean:
            ...

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __and__(self, other):
        return And(self, other)

    __rand__ = __and__

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __or__(self, other):
        return Or(self, other)

    __ror__ = __or__

    def __invert__(self):
        """Overloading for ~"""
        return Not(self)

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __rshift__(self, other):
        return Implies(self, other)

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __lshift__(self, other):
        return Implies(other, self)

    __rrshift__ = __lshift__
    __rlshift__ = __rshift__

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __xor__(self, other):
        return Xor(self, other)

    __rxor__ = __xor__

    def equals(self, other):
        """
        Returns ``True`` if the given formulas have the same truth table.
        For two formulas to be equal they must have the same literals.

        Examples
        ========

        >>> from sympy.abc import A, B, C
        >>> from sympy import And, Or, Not
        >>> (A >> B).equals(~B >> ~A)
        True
        >>> Not(And(A, B, C)).equals(And(Not(A), Not(B), Not(C)))
        False
        >>> Not(And(A, Not(A))).equals(Or(B, Not(B)))
        False

        """
        from sympy.logic.inference import satisfiable
        from sympy.core.relational import Relational

        if self.has(Relational) or other.has(Relational):
            raise NotImplementedError('handling of relationals')
        return self.atoms() == other.atoms() and \
            not satisfiable(Not(Equivalent(self, other)))

    def to_nnf(self, simplify=True):
        # override where necessary
        return self

    def as_set(self):
        """
        Rewrites Boolean expression in terms of real sets.

        Examples
        ========

        >>> from sympy import Symbol, Eq, Or, And
        >>> x = Symbol('x', real=True)
        >>> Eq(x, 0).as_set()
        {0}
        >>> (x > 0).as_set()
        Interval.open(0, oo)
        >>> And(-2 < x, x < 2).as_set()
        Interval.open(-2, 2)
        >>> Or(x < -2, 2 < x).as_set()
        Union(Interval.open(-oo, -2), Interval.open(2, oo))

        """
        from sympy.calculus.util import periodicity
        from sympy.core.relational import Relational

        free = self.free_symbols
        if len(free) == 1:
            x = free.pop()
            if x.kind is NumberKind:
                reps = {}
                for r in self.atoms(Relational):
                    if periodicity(r, x) not in (0, None):
                        s = r._eval_as_set()
                        if s in (S.EmptySet, S.UniversalSet, S.Reals):
                            reps[r] = s.as_relational(x)
                            continue
                        raise NotImplementedError(filldedent('''
                            as_set is not implemented for relationals
                            with periodic solutions
                            '''))
                new = self.subs(reps)
                if new.func != self.func:
                    return new.as_set()  # restart with new obj
                else:
                    return new._eval_as_set()

            return self._eval_as_set()
        else:
            raise NotImplementedError("Sorry, as_set has not yet been"
                                      " implemented for multivariate"
                                      " expressions")

    @property
    def binary_symbols(self):
        from sympy.core.relational import Eq, Ne
        return set().union(*[i.binary_symbols for i in self.args
                           if i.is_Boolean or i.is_Symbol
                           or isinstance(i, (Eq, Ne))])

    def _eval_refine(self, assumptions):
        from sympy.assumptions import ask
        ret = ask(self, assumptions)
        if ret is True:
            return true
        elif ret is False:
            return false
        return None

class Symbol(AtomicExpr, Boolean): # type: ignore
    """
    Symbol class is used to create symbolic variables.

    Explanation
    ===========

    Symbolic variables are placeholders for mathematical symbols that can represent numbers, constants, or any other mathematical entities and can be used in mathematical expressions and to perform symbolic computations.

    Assumptions:

    commutative = True
    positive = True
    real = True
    imaginary = True
    complex = True
    complete list of more assumptions- :ref:`predicates`

    You can override the default assumptions in the constructor.

    Examples
    ========

    >>> from sympy import Symbol
    >>> x = Symbol("x", positive=True)
    >>> x.is_positive
    True
    >>> x.is_negative
    False

    passing in greek letters:

    >>> from sympy import Symbol
    >>> alpha = Symbol('alpha')
    >>> alpha #doctest: +SKIP
    α

    Trailing digits are automatically treated like subscripts of what precedes them in the name.
    General format to add subscript to a symbol :
    ``<var_name> = Symbol('<symbol_name>_<subscript>')``

    >>> from sympy import Symbol
    >>> alpha_i = Symbol('alpha_i')
    >>> alpha_i #doctest: +SKIP
    αᵢ

    Parameters
    ==========

    AtomicExpr: variable name
    Boolean: Assumption with a boolean value(True or False)
    """

    is_comparable = False

    __slots__ = ('name', '_assumptions_orig', '_assumptions0')

    name: str

    is_Symbol = True
    is_symbol = True

    @property
    def kind(self):
        if self.is_commutative:
            return NumberKind
        return UndefinedKind

    @property
    def _diff_wrt(self):
        """Allow derivatives wrt Symbols.

        Examples
        ========

        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> x._diff_wrt
        True
        """
        return True

    @staticmethod
    def _sanitize(assumptions, obj=None):
        """Remove None, convert values to bool, check commutativity *in place*.
        """

        # be strict about commutativity: cannot be None
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        if is_commutative is None:
            whose = '%s ' % obj.__name__ if obj else ''
            raise ValueError(
                '%scommutativity must be True or False.' % whose)

        # sanitize other assumptions so 1 -> True and 0 -> False
        for key in list(assumptions.keys()):
            v = assumptions[key]
            if v is None:
                assumptions.pop(key)
                continue
            assumptions[key] = bool(v)

    def _merge(self, assumptions):
        base = self.assumptions0
        for k in set(assumptions) & set(base):
            if assumptions[k] != base[k]:
                raise ValueError(filldedent('''
                    non-matching assumptions for %s: existing value
                    is %s and new value is %s''' % (
                    k, base[k], assumptions[k])))
        base.update(assumptions)
        return base

    def __new__(cls, name, **assumptions):
        """Symbols are identified by name and assumptions::

        >>> from sympy import Symbol
        >>> Symbol("x") == Symbol("x")
        True
        >>> Symbol("x", real=True) == Symbol("x", real=False)
        False

        """
        cls._sanitize(assumptions, cls)
        return Symbol.__xnew_cached_(cls, name, **assumptions)


    @staticmethod
    @cacheit
    def _canonical_assumptions(**assumptions):
        # This is retained purely so that srepr can include commutative=True if
        # that was explicitly specified but not if it was not. Ideally srepr
        # should not distinguish these cases because the symbols otherwise
        # compare equal and are considered equivalent.
        #
        # See https://github.com/sympy/sympy/issues/8873
        #
        assumptions_orig = assumptions.copy()

        # The only assumption that is assumed by default is comutative=True:
        assumptions.setdefault('commutative', True)

        assumptions_kb = StdFactKB(assumptions)
        assumptions0 = dict(assumptions_kb)

        return assumptions_kb, assumptions_orig, assumptions0

    @staticmethod
    def __xnew__(cls, name, **assumptions):  # never cached (e.g. dummy)
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))


        obj = Expr.__new__(cls)
        obj.name = name

        assumptions_kb, assumptions_orig, assumptions0 = Symbol._canonical_assumptions(**assumptions)

        obj._assumptions = assumptions_kb
        obj._assumptions_orig = assumptions_orig
        obj._assumptions0 = tuple(sorted(assumptions0.items()))

        # The three assumptions dicts are all a little different:
        #
        #   >>> from sympy import Symbol
        #   >>> x = Symbol('x', finite=True)
        #   >>> x.is_positive  # query an assumption
        #   >>> x._assumptions
        #   {'finite': True, 'infinite': False, 'commutative': True, 'positive': None}
        #   >>> x._assumptions0
        #   {'finite': True, 'infinite': False, 'commutative': True}
        #   >>> x._assumptions_orig
        #   {'finite': True}
        #
        # Two symbols with the same name are equal if their _assumptions0 are
        # the same. Arguably it should be _assumptions_orig that is being
        # compared because that is more transparent to the user (it is
        # what was passed to the constructor modulo changes made by _sanitize).

        return obj

    @staticmethod
    @cacheit
    def __xnew_cached_(cls, name, **assumptions):  # symbols are always cached
        return Symbol.__xnew__(cls, name, **assumptions)

    def __getnewargs_ex__(self):
        return ((self.name,), self._assumptions_orig)

    # NOTE: __setstate__ is not needed for pickles created by __getnewargs_ex__
    # but was used before Symbol was changed to use __getnewargs_ex__ in v1.9.
    # Pickles created in previous SymPy versions will still need __setstate__
    # so that they can be unpickled in SymPy > v1.9.

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def _hashable_content(self):
        return (self.name,) + self._assumptions0

    def _eval_subs(self, old, new):
        if old.is_Pow:
            from sympy.core.power import Pow
            return Pow(self, S.One, evaluate=False)._eval_subs(old, new)

    def _eval_refine(self, assumptions):
        return self

    @property
    def assumptions0(self):
        return dict(self._assumptions0)

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (self.name,)), S.One.sort_key(), S.One

    def as_dummy(self):
        # only put commutativity in explicitly if it is False
        return Dummy(self.name) if self.is_commutative is not False \
            else Dummy(self.name, commutative=self.is_commutative)

    def as_real_imag(self, deep=True, **hints):
        if hints.get('ignore') == self:
            return None
        else:
            from sympy.functions.elementary.complexes import im, re
            return (re(self), im(self))

    def is_constant(self, *wrt, **flags):
        if not wrt:
            return False
        return self not in wrt

    @property
    def free_symbols(self):
        return {self}

    binary_symbols = free_symbols  # in this case, not always

    def as_set(self):
        return S.UniversalSet



def symbols(names, *, cls=Symbol, **args) -> Any:

    result = []

    if isinstance(names, str):
        marker = 0
        splitters = r'\,', r'\:', r'\ '
        literals: list[tuple[str, str]] = []
        for splitter in splitters:
            if splitter in names:
                while chr(marker) in names:
                    marker += 1
                lit_char = chr(marker)
                marker += 1
                names = names.replace(splitter, lit_char)
                literals.append((lit_char, splitter[1:]))
        def literal(s):
            if literals:
                for c, l in literals:
                    s = s.replace(c, l)
            return s

        names = names.strip()
        as_seq = names.endswith(',')
        if as_seq:
            names = names[:-1].rstrip()
        if not names:
            raise ValueError('no symbols given')

        # split on commas
        names = [n.strip() for n in names.split(',')]
        if not all(n for n in names):
            raise ValueError('missing symbol between commas')
        # split on spaces
        for i in range(len(names) - 1, -1, -1):
            names[i: i + 1] = names[i].split()

        seq = args.pop('seq', as_seq)

        for name in names:
            if not name:
                raise ValueError('missing symbol')

            if ':' not in name:
                symbol = cls(literal(name), **args)
                result.append(symbol)
                continue

            split: list[str] = _range.split(name)
            split_list: list[list[str]] = []
            # remove 1 layer of bounding parentheses around ranges
            for i in range(len(split) - 1):
                if i and ':' in split[i] and split[i] != ':' and \
                        split[i - 1].endswith('(') and \
                        split[i + 1].startswith(')'):
                    split[i - 1] = split[i - 1][:-1]
                    split[i + 1] = split[i + 1][1:]
            for s in split:
                if ':' in s:
                    if s.endswith(':'):
                        raise ValueError('missing end range')
                    a, b = s.split(':')
                    if b[-1] in string.digits:
                        a_i = 0 if not a else int(a)
                        b_i = int(b)
                        split_list.append([str(c) for c in range(a_i, b_i)])
                    else:
                        a = a or 'a'
                        split_list.append([string.ascii_letters[c] for c in range(
                            string.ascii_letters.index(a),
                            string.ascii_letters.index(b) + 1)])  # inclusive
                    if not split_list[-1]:
                        break
                else:
                    split_list.append([s])
            else:
                seq = True
                if len(split_list) == 1:
                    names = split_list[0]
                else:
                    names = [''.join(s) for s in product(*split_list)]
                if literals:
                    result.extend([cls(literal(s), **args) for s in names])
                else:
                    result.extend([cls(s, **args) for s in names])

        if not seq and len(result) <= 1:
            if not result:
                return ()
            return result[0]

        return tuple(result)
    else:
        for name in names:
            result.append(symbols(name, cls=cls, **args))

        return type(names)(result)

symbols('x')
