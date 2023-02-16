"""Module. Implements useful functions and classes for managing SU2 data.

"""
import numpy as np
import mpi

__all__ = [
    'Model',
    'ModelGrid',
    'ModelMarker',
    'preprocess_solver'
]


class Model:
    """Dataclass. SU2 model data structure.

    """
    def __init__(self, name, ndim, ids, domain):
        # model name
        self._name = name

        # model dimensions
        if ndim not in (2, 3):
            raise ValueError(f'ndim: Expected 2 or 3, got {ndim}')

        self._ndim = ndim

        # model grid
        self._grid = ModelGrid(self, ids, domain)

    def __repr__(self):
        # string representation
        return f'{self.__class__.__name__}({self.name}, grid={self.grid})'

    @property
    def name(self):
        """Property. The name of the Model.

        """
        return self._name

    @property
    def grid(self):
        """Property. Model Grid data structure.

        """
        return self._grid

    @property
    def ndim(self):
        """Property. The number of dimensions on the Model.

        """
        return self._ndim

    @property
    def nvar(self):
        """Property. The number of conservative variables on the Model.

        """
        return self._ndim + 2

    @property
    def nqoi(self):
        """Property. The number of primitive variables on the Model.

        """
        return 7


class ModelGrid:
    """Dataclass. SU2 grid data structure.

    """
    def __init__(self, parent, ids, domain):
        # store reference to parent
        if not isinstance(parent, Model):
            raise TypeError(f"Expected {Model.__name__}, got {parent}")

        self._parent = parent

        # store calculated partitioning information
        self._partitions = mpi.MPIPartitions(ids, domain)

        # data sorting information (applies the gathered data)
        if mpi.isroot():
            current = np.array(self.partitions.ids, dtype=int)
            forward = np.argsort(current)
            reverse = np.argsort(forward)
        else:
            current = np.array([], dtype=int)
            forward = np.array([], dtype=int)
            reverse = np.array([], dtype=int)

        self._forward = forward
        self._reverse = reverse

        self._data = {
            # mesh data
            'ids': current,
            'color': None,
            'coordinates': None,
            'displacements': None,
        }

        # dictionary of grid markers
        self._markers = {}

    def __repr__(self):
        # string representation
        return f'{self.__class__.__name__}({self.name}, size={self.size}, ndim={self.ndim}, markers={self.markers})'

    @property
    def parent(self):
        """Property. The parent Model of the Grid.

        """
        return self._parent

    @property
    def name(self):
        """Property. Name of the Grid.

        """
        return self.parent.name

    @property
    def partitions(self):
        """Property. List of Grid partitions.

        """
        return self._partitions

    @property
    def partition(self):
        """Property. Current Grid partition.

        """
        return self._partitions.current

    @property
    def size(self):
        """Property. Size of the Grid.

        """
        return self._partitions.size

    @property
    def ndim(self):
        """Property. Size of the Grid.

        """
        return self.parent.ndim

    @property
    def nvar(self):
        """Property. Size of the Grid.

        """
        return self.parent.nvar

    @property
    def nqoi(self):
        """Property. Size of the Grid.

        """
        return self.parent.nqoi

    @property
    def markers(self):
        """Property. Dictionary of Grid Markers.

        """
        return self._markers

    def marker(self, name):
        """Method. Return a Marker by name.

        """
        try:
            return self._markers[name]
        except KeyError as error:
            raise KeyError(f"Marker '{name}' is not defined; try one of {list(self.markers)}") from error

    def tag(self, name, index, ids, domain):
        """Method. Define a new Marker on the Grid.

        """
        if name in self._markers:
            raise KeyError(f"Marker '{name}' is already defined !")

        self._markers[name] = ModelMarker(self, name, index, ids, domain)

    def reorder(self, ids):
        """Method. Re-order the internal data and partitioning information.

        """
        mapping = self.partitions.reorder(ids)

        # apply mapping to the sorting information
        if mpi.isroot():
            current = self.partitions.ids

            forward = np.argsort(current)
            reverse = np.argsort(forward)
        else:
            forward = np.array([], dtype=int)
            reverse = np.array([], dtype=int)

        self._forward = forward
        self._reverse = reverse

        if mpi.isroot():
            # apply mapping to the grid data
            for name, data in self._data.items():
                if data is None:
                    continue

                self._data[name] = np.ascontiguousarray(data[mapping])

    def sort(self, values, reverse=False):
        """Function. Sort the values by increasing Grid IDs.

        """
        if values is None:
            return None
        if reverse:
            return values[self._reverse]
        else:
            return values[self._forward]


class ModelMarker:
    """Dataclass. SU2 marker metadata.

    """
    def __init__(self, parent, name, index, ids, domain):

        # store reference to parent
        if not isinstance(parent, ModelGrid):
            raise TypeError(f"Expected {ModelGrid.__name__}, got {parent}")

        self._parent = parent

        # marker name and index
        self._name = name
        self._index = index

        # store calculated partitioning information
        self._partitions = mpi.MPIPartitions(ids, domain)

        # marker mapping indices (to volume data)
        self._total = [
            self.parent.partitions.ids.index(_id) for _id in self.partitions.ids
        ]
        self._local = [
            self.parent.partition.ids.index(_id) for _id in self.partition.ids
        ]

        # data sorting information (applies the gathered data)
        if mpi.isroot():
            current = self.partitions.ids

            forward = np.argsort(current)
            reverse = np.argsort(forward)
        else:
            forward = np.array([], dtype=int)
            reverse = np.array([], dtype=int)

        self._forward = forward
        self._reverse = reverse

    def __repr__(self):
        # string representation
        return f'{self.__class__.__name__}({self.name}, size={self.size}, ndim={self.ndim})'

    @property
    def parent(self):
        """Property. The parent Grid of the Marker.

        """
        return self._parent

    @property
    def name(self):
        """Property. The tag (i.e. name) of the Marker.

        """
        return self._name

    @property
    def index(self):
        """Property. The index of the Marker on the current rank.

        """
        return self._index

    @property
    def local(self):
        """Property. The local indices of the Marker data in the Model.


        """
        return self._local

    @property
    def total(self):
        """Property. The total indices of the Marker data in the Model.

        """
        return self._total

    @property
    def partitions(self):
        """Property. List of Marker partitions.

        """
        return self._partitions

    @property
    def partition(self):
        """Property. Current Marker partition.

        """
        return self._partitions.current

    @property
    def size(self):
        """Property. Size of the Marker.

        """
        return self._partitions.size

    @property
    def ndim(self):
        """Property. Dimensions of the Marker.

        """
        return self.parent.ndim

    @property
    def nvar(self):
        """Property. Size of the Grid.

        """
        return self.parent.nvar

    @property
    def nqoi(self):
        """Property. Size of the Grid.

        """
        return self.parent.nqoi

    def sort(self, values, reverse=False):
        """Function. Sort the values by increasing Grid IDs.

        """
        if values is None:
            return None
        if reverse:
            return values[self._reverse]
        else:
            return values[self._forward]

    def reorder(self, ids):
        """Method. Re-order the Marker data and partitioning information.

        """
        mapping = self.partitions.reorder(ids)

        # marker mapping indices (to volume data)
        self._total = [
            self.parent.partitions.ids.index(_id) for _id in self.partitions.ids
        ]
        self._local = [
            self.parent.partition.ids.index(_id) for _id in self.partition.ids
        ]

        # apply mapping to the sorting information
        if mpi.isroot():
            current = self.partitions.ids

            forward = np.argsort(current)
            reverse = np.argsort(forward)
        else:
            forward = np.array([], dtype=int)
            reverse = np.array([], dtype=int)

        self._forward = forward
        self._reverse = reverse


def preprocess_solver(solver, markers):
    ndim = solver.GetNumberDimensions()
    print(f'* Number of dimensions: {ndim}')

    size = solver.GetNumberVertices()
    halo = solver.GetNumberHaloVertices()
    real = size - halo

    for rank in range(mpi.SIZE):
        print(f'* ----- RANK {rank} ---')
        print(f'* Number of vertices           : {size}')
        print(f'* Number of vertices (internal): {real}')
        print(f'* Number of vertices (boundary): {halo}')
        mpi.barrier()

    # mesh vertex identifiers [i1, i2, ... ]
    ids = solver.GetVertexIDs()

    # mesh vertex partitioning [h1]
    domain = solver.GetDomain()

    # initialize model
    name = 'RAPTOR'
    model = Model(name, ndim, ids, domain)

    # list of all boundary markers
    tags = solver.GetMarkerIndices()

    if mpi.isparallel():
        tags_global = set(tag for rank in mpi.COMM.allgather(tags) for tag in rank)
    else:
        tags_global = set(tags)

    # list of all deformable boundary markers
    tags_deform = solver.GetDeformableMarkerTags()

    for tag in markers:
        print(f'Processing model marker data for {tag}...')

        if tag not in tags_global:
            raise ValueError(f'Marker {tag} is not defined!')
        if tag not in tags_deform:
            raise ValueError(f'Marker {tag} is not deformable!')

        # get marker index (it is None if not defined on this rank)
        index = tags.get(tag)

        for rank in range(mpi.SIZE):
            print(f'* ----- RANK {rank} ---')
            print(f'* Marker index: {index}')
            mpi.barrier()

        # get marker sizes
        if index is None:
            size = 0
            halo = 0
        else:
            size = solver.GetNumberMarkerVertices(index)
            halo = solver.GetNumberMarkerHaloVertices(index)

        real = size - halo

        for rank in range(mpi.SIZE):
            print(f'* ----- RANK {rank} ---')
            print(f'* Number of vertices           : {size}')
            print(f'* Number of vertices (internal): {real}')
            print(f'* Number of vertices (boundary): {halo}')
            mpi.barrier()

        # mesh vertex identifiers [i1, i2, ... ]
        if index is None:
            ids = []
        else:
            ids = solver.GetMarkerVertexIDs(index)

        if index is None:
            domain = []
        else:
            domain = solver.GetMarkerDomain(index)

        # initialize the marker partitioning information
        model.grid.tag(tag, index, ids, domain)

    # preprocess flow solver
    solver.Preprocess(0)
    ids = solver.GetVertexIDs()

    model.grid.reorder(ids)

    # apply mapping to the markers
    for name, marker in model.grid.markers.items():
        if marker.index is None:
            ids = ()
        else:
            ids = solver.GetMarkerVertexIDs(marker.index)

        marker.reorder(ids)

    return model
