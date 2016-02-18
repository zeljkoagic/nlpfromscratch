class Arc:
    __slots__ = ['u', 'v', 'u_pos', 'v_pos', 'weight', 'flow_var', 'edge_var']

    def __init__(self, u: int, v: int, u_pos: int, v_pos: int, weight: float):
        self.u = u
        self.v = v
        self.u_pos = u_pos
        self.v_pos = v_pos
        self.weight = weight

    def __repr__(self):
        return "{}_{}_{}_{}".format(self.u, self.v, self.u_pos, self.v_pos)
