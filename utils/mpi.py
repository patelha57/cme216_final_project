"""Module. Implements and configures the interface to MPI.

"""
import functools
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

    COMM = None
    SIZE = 1
    RANK = 0
    ROOT = 0

else:
    COMM = MPI.COMM_WORLD
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    ROOT = 0


__all__ = [
    'COMM',
    'SIZE',
    'RANK',
    'ROOT',
    'isroot',
    'ifroot',
    'isrank',
    'ifrank',
    'isparallel',
    'ifparallel',
    'barrier',
    'broadcast',
    'allsum',
    'allgather',
    'gather',
    'scatter',
    'gather_partitions',
    'scatter_partitions',
    'Partition'
]


def isrank(rank):
    """Function. Return True on the root MPI rank, False elsewhere.

    """
    if rank is None:
        rank = ROOT

    return RANK == rank


def ifrank(function):
    """Function. Decorate a function to operate only on the given rank(s).

    """
    @functools.wraps(function)
    def conditional(*args, rank=None, **kwargs):
        """Internal function. Check if a function is run on a requested rank.

        """
        if isrank(rank):
            return function(*args, **kwargs)
        else:
            return None

    return conditional


def isroot():
    """Function. Return True on the root MPI rank, False elsewhere.

    """
    return RANK == ROOT


def ifroot(function):
    """Function. Decorate a function to operate only on the root rank.

    """
    @functools.wraps(function)
    def conditional(*args, **kwargs):
        """Internal function. Check if a function is run on the root rank.

        """
        if isroot():
            return function(*args, **kwargs)
        else:
            return None

    return conditional


def isparallel():
    """Function. Return True if more than one MPI rank is available.

    """
    return SIZE > 1


def ifparallel(function):
    """Function. Decorate a function to operate only on parallel runs.

    """
    @functools.wraps(function)
    def conditional(*args, **kwargs):
        """Internal function. Check if a function is run on a parallel run.

        """
        if isparallel():
            return function(*args, **kwargs)
        else:
            return None

    return conditional


def barrier():
    """Function. MPI synchronization barrier.

    """
    COMM.Barrier()


def broadcast(value):
    """Function. Broadcast a value from root to all ranks.

    """
    return COMM.bcast(value, root=ROOT)


def allsum(value):
    """Function. Sum a value over all ranks.

    """
    return COMM.allreduce(value, op=MPI.SUM)


def allgather(value):
    """Function. Gather a value over all ranks.

    """
    return COMM.allgather(value)


def gather(values):
    """Function. Gather partitioned values to root rank.

    """
    if not isparallel():
        return values

    return COMM.gather(values, root=ROOT)


def scatter(values):
    """Function. Scatter partitioned values to root rank.

    """
    if not isparallel():
        return values

    return COMM.scatter(values, root=ROOT)


def gather_partitions(values, partitions=None):
    """Function. Gather SU2 partitions to root rank.

    """
    if not isparallel():
        return values

    if partitions is None:
        raise ValueError(f"No partitions were provided!")
    if not len(partitions) == SIZE:
        raise ValueError(f"Expected {SIZE} partitions, got {len(partitions)}")

    for partition in partitions:
        if not isinstance(partition, Partition):
            raise ValueError(f"Expected {Partition}, got {type(partition)}")

    # get local partition
    partition = partitions[RANK]

    # Prepare metadata
    size = sum(partition.internal.size for partition in partitions)
    dims = values.shape[1:]
    data = values.dtype

    # prepare receive buffer
    if isroot():
        recv = np.empty((size, *dims), dtype=data)
    else:
        recv = np.empty((0, *dims), dtype=data)

    # prepare send buffer (must be contiguous array of internal data)
    send = np.ascontiguousarray(values[partition.internal])

    # Non-blocking send the data on all ranks ...
    request = COMM.Isend(send, dest=ROOT, tag=0)

    # blocking receive the data on root
    if isroot():
        shift = 0
        for rank, partition in enumerate(partitions):
            COMM.Recv(recv[shift:shift + partition.internal.size], source=rank, tag=0)

            shift += partition.internal.size

    request.wait()
    return recv


def scatter_partitions(values, partitions=None):
    """Function. Scatter SU2 partitions to root rank.

    """
    if not isparallel():
        return values

    if partitions is None:
        raise ValueError(f"No partitions were provided!")
    if not len(partitions) == SIZE:
        raise ValueError(f"Expected {SIZE} partitions, got {len(partitions)}")

    for partition in partitions:
        if not isinstance(partition, Partition):
            raise ValueError(f"Expected {Partition}, got {type(partition)}")

    # get local partition
    partition = partitions[RANK]

    # prepare metadata
    size = sum(partition.internal.size for partition in partitions)

    if isroot():
        dims = values.shape[1:]
        data = values.dtype
    else:
        dims = None
        data = None

    dims = broadcast(dims)
    data = broadcast(data)

    if isroot() and not values.shape[0] == size:
        raise ValueError(f"Expected size {size}, got {values.shape[0]}")

    # prepare send buffer (must be contiguous array of internal data)
    if isroot():
        send = np.asarray(values)
    else:
        send = None

    recv_internal = np.empty((partition.internal.size, *dims), dtype=data)
    recv_boundary = np.empty((partition.boundary.size, *dims), dtype=data)

    # non-blocking receive the data on all ranks ...
    request_internal = COMM.Irecv(recv_internal, source=ROOT, tag=1)
    request_boundary = COMM.Irecv(recv_boundary, source=ROOT, tag=2)

    # blocking send the data on root
    if isroot():
        shift = 0
        for rank, partition in enumerate(partitions):
            send_internal = np.ascontiguousarray(send[shift:shift + partition.internal.size])
            send_boundary = np.ascontiguousarray(send[partition.exchange])

            COMM.Send(send_internal, dest=rank, tag=1)
            COMM.Send(send_boundary, dest=rank, tag=2)

            shift += partition.internal.size

    request_internal.wait()
    request_boundary.wait()

    # fill internal and boundary data
    recv = np.empty((partition.size, *dims), dtype=data)
    recv[partition.internal] = recv_internal
    recv[partition.boundary] = recv_boundary

    return recv


class Partition:
    """Dataclass. MPI partition information.

    """

    def __init__(self, internal, boundary, exchange):
        # array of internal (aka "physical" or "domain") vertices
        self._internal = np.asarray(internal)

        # array of boundary (aka "ghost" or "halo") vertices
        self._boundary = np.asarray(boundary)

        # dictionary mapping boundary indices in local partition to total indices
        self._exchange = exchange

    @property
    def size(self):
        """Property. Combined size of internal and boundary vertices.

        """
        return self._internal.size + self._boundary.size

    @property
    def internal(self):
        """Property. Array of internal (i.e. physical, domain) indices.

        """
        return self._internal

    @property
    def boundary(self):
        """Property. Array of boundary (i.e. ghost, halo) indices.

        """
        return self._boundary

    @property
    def exchange(self):
        """Property. Mapping of local boundary indices to global indices.

        """
        return self._exchange
