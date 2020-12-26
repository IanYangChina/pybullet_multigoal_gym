import random as R


class StepDemonstrator(object):
    def __init__(self, demonstrations, stick_with_final_goal=True, seed=0):
        R.seed(seed)
        self.demonstrations = demonstrations
        self.demon_num = len(self.demonstrations)
        self.demon_ind = 0
        self.current_goal = -1
        self.current_final_goal = 0
        self.stick_with_final_goal = stick_with_final_goal

    def get_next_goal(self):
        if self.stick_with_final_goal and (self.current_goal != -1):
            if self.demonstrations[self.demon_ind][self.current_goal] == self.demonstrations[self.demon_ind][-1]:
                return self.demonstrations[self.demon_ind][self.current_goal]
        self.current_goal = (self.current_goal+1) % len(self.demonstrations[self.demon_ind])
        return self.demonstrations[self.demon_ind][self.current_goal]

    def manual_reset(self, demon_ind):
        self.current_goal = -1
        self.demon_ind = demon_ind
        self.current_final_goal = self.demonstrations[self.demon_ind][-1]

    def reset_with_the_last_sub_goal_index(self, ind):
        self.current_goal = -1
        for i in range(self.demon_num):
            if self.demonstrations[i][-1] == ind:
                self.demon_ind = i
                break
        self.current_final_goal = self.demonstrations[self.demon_ind][-1]