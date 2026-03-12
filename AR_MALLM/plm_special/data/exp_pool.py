
class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self):
        self.agent_ids = [] #########
        self.pre_rs = [] ###########
        self.states = []
        self.actions = []
        self.rewards = []
        self.agent_actives = [] #########

    def add(self, agent_id, pre_r, state, action, reward, agent_active):
        self.agent_ids.append(agent_id)  #########
        self.pre_rs.append(pre_r)  ###########
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.agent_actives.append(agent_active) ##############

    def __len__(self):
        return len(self.agent_ids)

