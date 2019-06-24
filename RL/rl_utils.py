"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import numpy as np
import threading, random
import tensorflow as tf


class memory_pooling(object):
    """a class offer the APIs of experience replay in reinforcement learning
    Example:
        memorys = memory_pooling(maxlen=100)

    for i in range(300):
        state_current = [1, 2, 3, 4]
        action = [1., 2.3, 4.6]
        state_next = [2, 3, 5, 9]
        r = 2.3
        whether_end = False

        memorys.put([state_current, action, state_next, r, whether_end])
    pass
    """
    def __init__(self, maxlen=500):
        self.lock = threading.Lock()
        self.memory = []
        self.maxlen = maxlen


    def put(self, memory):
        """put the a memory to memory list
        Args:
        """
        self.lock.acquire()
        ## put memory, could not be interupted by other threads
        if len(self.memory) >= self.maxlen:
            self.memory.pop(0) ## del the earliest memory
            self.memory.append(memory) ## put the newest memory
        else:
            self.memory.append(memory)
        self.lock.release()


    def capacity_bigger_than(self, val):
        """judge whether the memory pooling capacity is bigger that a val
        Args:
            val: normally, is a int represents the batch size
        Return:
            bool var.
        """
        if len(self.memory) >= val:
            return True
        else:
            return  False


    def get(self, batch_size):
        """get some memory from memory pooling
        Args:
            batch_size: an int
        Return:
            if success, return batch memories.
            elsewise, return None
        """
        if len(self.memory) >= batch_size:
            self.lock.acquire()
            memorys = random.sample(self.memory, batch_size)
            self.lock.release()
            return memorys
        else:
            return None
        pass


class balance_memory_pooling():
    def __init__(self, max_capacity, n_class):
        self.balance_memory = []
        self.max_capacity = max_capacity
        self.n_class = n_class
        for i in range(n_class):
            self.balance_memory.append([])

        self.whether_max_capacity = False


    def put(self, class_index, memory):
        assert class_index < len(self.balance_memory)

        if self.__total_capacity() >= self.max_capacity:
            self.__del_memory_of_max_len()
            self.balance_memory[class_index].append(memory)
        else:
            self.balance_memory[class_index].append(memory)


    def __total_capacity(self):
        l = 0
        for memorys in self.balance_memory:
            l += len(memorys)
        return l


    def __del_memory_of_max_len(self):
        l = []
        for memorys in self.balance_memory:
            l.append(len(memorys))

        index = int(np.argmax(np.array(l)))
        self.balance_memory[index].pop(0)


    def get_propotion(self):
        l = []
        for memorys in self.balance_memory:
            l.append(len(memorys))

        propotion = np.array(l)/ self.__total_capacity()
        return propotion


    def get(self, batch_size):
        m = []
        for memorys in self.balance_memory:
            m += memorys

        return random.sample(m, batch_size)


    def capacity_bigger_than(self, val):
        """judge whether the memory pooling capacity is bigger that a val
        Args:
            val: normally, is a int represents the batch size
        Return:
            bool var.
        """
        if self.__total_capacity() >= val:
            return True
        else:
            return False

    def is_balance(self):
        propotion = float(np.max(np.array(self.get_propotion())))
        if propotion < 0.15:
            return True
        else:
            return False


def normalize_rewards(rewards):
    """normalize the rewards
    Args:
        rewards: a list of reward.
    Return:
        rewards after normalization
    """
    r = (np.array(rewards) - np.mean(rewards))/np.std(rewards)
    return r


def replace(memorys, norm_rewards, reward_index=3):
    """replace the raw rewards in memorys with the rewards after normalizaiton
    Args:
        memorys: a list of memory, the elemt means [state_current, action, state_next, reward, end]
        norm_rewards: a list of rewards after normalization.
        reward_index: the raw reward position in memories' item, default is 3

    Example:
        ## init a memorys pooling
        memorys = memory_pooling(maxlen=100)

        for i in range(300):
            memorys.put([i,i,i,i,i])

        ## get 20 memorys
        memorys_ = memorys.get(batch_size=20)

        ## calculate the norm_rewards and replace raw rewards with them.
        raw_rewards = [m[3] for m in memorys_]
        r = normalize_rewards(raw_rewards)
        replace(memorys_, r)

        ## now, the memorys_'s rewards are after nomalization
    """
    for i, nr in enumerate(norm_rewards):
        memorys[i][reward_index] = nr


def copy_a2b(vars_a, vars_b):
    """creat a copy ops which copy the vars_a to vars_b
    Args:
        vars_a: a list of tensor vars
        var_b: a list of tensor vars
    Return:
        a list of copy operation
    """
    assert len(vars_a)==len(vars_b)
    copy_ops = [var_b.assign(var_a) for var_a, var_b in zip(vars_a, vars_b)]
    copy_ops = tf.group(*copy_ops)
    return copy_ops


def soft_copy_a2b(vars_a, vars_b, tau=1e-3):
    """creat a soft copy ops which soft copy the vars_a to vras_b
    vars_b = tau*vars_a + (1-tau)*vars_b

    Args:
        vars_a: a list of tensor vars
        var_b: a list of tensor vars
    Return:
        a list of soft copy operation
    """
    assert len(vars_a) == len(vars_b)
    copy_ops = [var_b.assign(tau*var_a + (1-tau)*var_b) for var_a, var_b in zip(vars_a, vars_b)]
    copy_ops = tf.group(*copy_ops)
    return copy_ops


if __name__ == '__main__':
    memorys = memory_pooling(maxlen=100)

    for i in range(300):
        memorys.put([i,i,i,i,i])

    memorys_ = memorys.get(batch_size=20)


    raw_rewards = [m[3] for m in memorys_]

    r = normalize_rewards(raw_rewards)
    replace(memorys_, r)

    pass