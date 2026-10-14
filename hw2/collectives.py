import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    right = (rank + 1) % world_size
    left = (rank - 1) % world_size

    # На шаге step отправляем правому соседу частичную сумму куска (rank - step - 1),
    # от левого получаем кусок (rank - step - 2) и добавляем к своей копии.
    # После world_size - 1 шагов в chunks[rank] лежит сумма со всех процессов.
    for step in range(world_size - 1):
        send_idx = (rank - step - 1) % world_size
        recv_idx = (rank - step - 2) % world_size
        received = torch.empty_like(chunks[recv_idx])
        send = dist.isend(chunks[send_idx], dst=right)
        receive = dist.irecv(received, src=left)
        receive.wait()
        send.wait()
        chunks[recv_idx].add_(received)

    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    right = (rank + 1) % world_size
    left = (rank - 1) % world_size

    # На шаге step пересылаем дальше шард, который получили на прошлом шаге
    # (на нулевом шаге — свой). За world_size - 1 шагов у всех есть все шарды.
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size
        send = dist.isend(chunks[send_idx], dst=right)
        receive = dist.irecv(chunks[recv_idx], src=left)
        receive.wait()
        send.wait()

    return torch.cat(chunks)


def ring_all_reduce(tensor):
    # reduce-scatter даёт каждому процессу свой кусок суммы, all-gather собирает их обратно.
    return ring_all_gather(ring_reduce_scatter(tensor))
