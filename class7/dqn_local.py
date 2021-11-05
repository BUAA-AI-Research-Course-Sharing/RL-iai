# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #7: The Maze Decorator
import torch
import torch.nn as nn
import torch.nn.functional as F

# try:
    # from malmo import MalmoPython
# except:
    # import MalmoPython
import cv2
import glog as log
import json
import numpy as np
# from priority_dict import priorityDictionary as PQ

# TODO: 
MAZE_DATA_PATH = "./map.json"

# NOTE: hyper-parameters listed as below:
BATCH_SIZE = 32
LR = 0.01  # 学习率
EPSILON = 0.8               # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
GAMMA = 0.9  # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
TARGET_REPLACE_ITER = 4  # Q现实网络的更新频率100次循环更新一次
MEMORY_CAPACITY = 2000  # 记忆库大小
N_ACTIONS = 4  # 棋子的动作0，1，2，3
N_STATES = 1


def trans_torch(list1):
    list1 = np.array(list1)
    l1 = np.where(list1 == 1, 1, 0)
    l2 = np.where(list1 == 2, 1, 0)
    l3 = np.where(list1 == 3, 1, 0)
    b = np.array([l1, l2, l3])
    return b


# 神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(3, 16, 3, 1, 0)
        self.c2 = nn.Conv2d(16, 25, 3, 1, 0)
        self.f1 = nn.Linear(25, 16)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2 = nn.Linear(16, 4)
        self.f2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # print(x.shape)
        x = self.c1(x)
        # print(x.shape)
        x = F.relu(x)
        # print(x.shape)
        x = self.c2(x)
        # print(x.shape)
        x = F.relu(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.f1(x)
        x = F.relu(x)
        action = self.f2(x)
        return action


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # DQN需要使用两个神经网络
        # eval为Q估计神经网络 target为Q现实神经网络
        self.learn_step_counter = 0  # 用于 target 更新计时，100次更新一次
        self.memory_counter = 0  # 记忆库记数
        self.memory = list(np.zeros((MEMORY_CAPACITY, 4)))  # 初始化记忆库用numpy生成一个(2000,4)大小的全0矩阵，
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample,x为场景
        if np.random.uniform() < EPSILON:  # 选最优动作
            actions_value = self.eval_net.forward(x)  # 将场景输入Q估计神经网络
            # torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
            action = torch.max(actions_value, 1)[1].data.numpy()  # 返回动作最大值
        else:  # 选随机动作
            action = np.array([np.random.randint(0, N_ACTIONS)])  # 比如np.random.randint(0,2)是选择1或0
        return action

    def store_transition(self, s, a, r, s_):
        # 如果记忆库满了, 就覆盖老数据，2000次覆盖一次
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = [s, a, r, s_]
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新,每100次
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 将所有的eval_net里面的参数复制到target_net里面
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        # 从2000以内选择32个数据标签
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_s = []
        b_a = []
        b_r = []
        b_s_ = []
        for i in sample_index:
            b_s.append(self.memory[i][0])
            b_a.append(np.array(self.memory[i][1], dtype=np.int32))
            b_r.append(np.array([self.memory[i][2]], dtype=np.int32))
            b_s_.append(self.memory[i][3])
        b_s = torch.FloatTensor(b_s)  # 取出s
        b_a = torch.LongTensor(b_a)  # 取出a
        b_r = torch.FloatTensor(b_r)  # 取出r
        b_s_ = torch.FloatTensor(b_s_)  # 取出s_
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach Q现实
        q_target = b_r + GAMMA * q_next.max(1)[0].reshape(BATCH_SIZE, 1)  # shape (batch, 1) DQL核心公式
        # print(np.argmax(q_next,axis=1).reshape(1,BATCH_SIZE))
        # print(q_eval,q_target)
        loss = self.loss_func(q_eval, q_target)  # 计算误差
        # 计算, 更新 eval net
        # print(loss)
        self.optimizer.zero_grad()  #
        loss.backward()  # 反向传递
        self.optimizer.step()
        return loss


class Env():
    def __init__(self, start, end):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.start = start
        self.end = end

    # 建立虚拟环境
    def start_env(self, start_states):
        self.y1 = self.start % 21 - 10
        self.x1 = self.start - (self.y1 + 10) * 21 - 7
        self.migong = start_states.copy()
        self.end_game = 0
        # print(self.migong)
        return self.migong

    def display(self, info=''):
        self.display1 = np.ones((300, 300, 3), dtype=np.uint8)
        self.display1 = np.array(np.where(self.display1 == 1, 255, 0), dtype=np.uint8)
        for i in range(5):
            cv2.line(self.display1, (i * 60, 0), (i * 60, 300), (0, 0, 0), 1)
            cv2.line(self.display1, (0, i * 60), (300, i * 60), (0, 0, 0), 1)
        for x in range(5):
            for y in range(5):
                if self.migong[y][x] == 1:
                    cv2.circle(self.display1, (x * 60 + 30, y * 60 + 30), 25, (255, 0, 0), -1)
                if self.migong[y][x] == 2:
                    cv2.circle(self.display1, (x * 60 + 30, y * 60 + 30), 25, (0, 255, 0), -1)
                if self.migong[y][x] == 3:
                    cv2.circle(self.display1, (x * 60 + 30, y * 60 + 30), 25, (0, 0, 255), -1)

        cv2.imshow(info, self.display1)
        cv2.waitKey(100)


    def step(self, action, loc):
        r = 0
        # ['u'0, 'd'1, 'l'2, 'r'3]
        if action == 0:
            if self.y1 == 0:
                r = -0.5
            elif loc - 21 in legal_states:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1 - 1][self.x1] = 1
                self.y1 -= 1
                loc -= 21
                if loc == end:
                    self.end_game = 2
                    r = 100
            else:
                self.end_game = 1
                r = -1

        if action == 1:
            if self.y1 == 4:
                r = -0.5
            elif loc + 21 in legal_states:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1 + 1][self.x1] = 1
                self.y1 += 1
                loc += 21
                if loc == end:
                    self.end_game = 2
                    r = 100
            else:
                self.end_game = 1
                r = -1
        if action == 2:
            if self.x1 == 0:
                r = -0.5
            elif loc - 1 in legal_states:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1][self.x1 - 1] = 1
                self.x1 -= 1
                loc -= 1
                if loc == end:
                    self.end_game = 2
                    r = 100
            else:
                self.end_game = 1
                r = -1
        if action == 3:
            if self.x1 == 4:
                r = -0.5
            elif loc + 1 in legal_states:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1][self.x1 + 1] = 1
                self.x1 += 1
                loc += 1
                if loc == end:
                    self.end_game = 2
                    r = 100
            else:
                self.end_game = 1
                r = -1
        # return self.migong
        return self.end_game, r, self.migong, loc



# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
"""
def GetMissionXML(seed, gp, size=10):
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Hello world!</Summary>
              </About>

            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                  <DrawingDecorator>
                    <DrawSphere x="-27" y="70" z="0" radius="30" type="air"/>
                  </DrawingDecorator>
                  <MazeDecorator>
                    <Seed>''' + str(seed) + '''</Seed>
                    <SizeAndPosition width="''' + str(size) + '''" length="''' + str(size) + '''" height="10" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
                    <StartBlock type="emerald_block" fixedToEdge="true"/>
                    <EndBlock type="redstone_block" fixedToEdge="true"/>
                    <PathBlock type="diamond_block"/>
                    <FloorBlock type="air"/>
                    <GapBlock type="air"/>
                    <GapProbability>''' + str(gp) + '''</GapProbability>
                    <AllowDiagonalMovement>false</AllowDiagonalMovement>
                  </MazeDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>CS175AwesomeMazeBot</Name>
                <AgentStart>
                    <Placement x="0.5" y="56.0" z="0.5" yaw="0"/>
                </AgentStart>
                <AgentHandlers>
                    <DiscreteMovementCommands/>
                    <AgentQuitFromTouchingBlockType>
                        <Block type="redstone_block"/>
                    </AgentQuitFromTouchingBlockType>
                    <ObservationFromGrid>
                      <Grid name="floorAll">
                        <min x="-10" y="-1" z="-10"/>
                        <max x="10" y="-1" z="10"/>
                      </Grid>
                  </ObservationFromGrid>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''
"""

def load_grid(mission_id):
    """
    Used the agent observation API to get a 21 X 21 grid box around the agent (the agent is in the middle).

    Args
        mission_id:    <object>    current mission id

    Returns
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)
    """
    with open(MAZE_DATA_PATH, 'r') as fp:
        maze_dict = json.load(fp)

    maze = maze_dict[str(mission_id)]
    del maze_dict

    return maze


def find_start_end(grid):
    """
    Finds the source and destination block indexes from the list.

    Args
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)

    Returns
        start: <int>   source block index in the list
        end:   <int>   destination block index in the list
    """
    start = grid.index(u'emerald_block')
    end = grid.index(u'redstone_block')
    return (start, end)
    # -------------------------------------


def extract_action_list_from_path(path_list):
    """
    Converts a block idx path to action list.

    Args
        path_list:  <list>  list of block idx from source block to dest block.

    Returns
        action_list: <list> list of string discrete action commands (e.g. ['movesouth 1', 'movewest 1', ...]
    """
    action_trans = {-21: 'movenorth 1', 21: 'movesouth 1', -1: 'movewest 1', 1: 'moveeast 1'}
    alist = []
    for i in range(len(path_list) - 1):
        curr_block, next_block = path_list[i:(i + 2)]
        if next_block == curr_block:
            continue
        alist.append(action_trans[next_block - curr_block])

    return alist


# 根据最终的qfunc得到最优路径
def get_shortest_path(qfunc, mission_id=None, display=False):
    # -------------------------------------
    # 根据最终得到的qfunc输出最优的路径
    # -------------------------------------
    EPSILON = 1 # set epsilon=1 for inference shortest path
    
    info = '' if mission_id is None\
        else f"Mission {mission_id}"

    state = start
    path = []
    path.append(state)
    maze_image = env.start_env(start_states)
    feat_maps = trans_torch(maze_image)
    # print(qfunc)

    if display:
        env.display(info)
    
    while state != end:
        a = dqn.choose_action(feat_maps)
        # self.end_game, r, self.migong, loc
        _, _, maze_image, state = env.step(a, state)
        
        if display:
            env.display(info)
        
        feat_maps = trans_torch(maze_image)
        path.append(state)

    cv2.destroyWindow(info)

    return path


# Create default Malmo objects:
# agent_host = MalmoPython.AgentHost()
# try:
    # agent_host.parse(sys.argv)
# except RuntimeError as e:
    # print('ERROR:', e)
    # print(agent_host.getUsage())
    # exit(1)
# if agent_host.receivedArgument("help"):
    # print(agent_host.getUsage())
    # exit(0)

# if agent_host.receivedArgument("test"):
    # num_repeats = 1
# else:
num_repeats = 5
    
dqn = DQN()  # 定义 DQN 系统
for mission_id in range(num_repeats):
    size = int(5)
    print("Size of maze:", size)
    # my_mission = MalmoPython.MissionSpec(GetMissionXML("0", 0.4 + float(i / 20.0), size), True)
    # my_mission_record = MalmoPython.MissionRecordSpec()
    # my_mission.requestVideo(800, 500)
    # my_mission.setViewpoint(1)
    # # Attempt to start a mission:
    # max_retries = 3
    # my_clients = MalmoPython.ClientPool()
    # my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    # for retry in range(max_retries):
        # try:
            # agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "%s-%d" % ('Moshe', i))
            # break
        # except RuntimeError as e:
            # if retry == max_retries - 1:
                # print("Error starting mission", (i + 1), ":", e)
                # exit(1)
            # else:
                # time.sleep(2)

    # Loop until mission starts:
    # print("Waiting for the mission", (mission_id + 1), "to start ", )
    # world_state = agent_host.getWorldState()
    # while not world_state.has_mission_begun:
        # # sys.stdout.write(".")
        # time.sleep(0.1)
        # world_state = agent_host.getWorldState()
        # for error in world_state.errors:
            # print("Error:", error.text)

    # print()
    print("Mission", (mission_id + 1), "running.")

    grid = load_grid(mission_id)
    start, end = find_start_end(grid)
    
    start_states = []
    legal_states = []
    legal_blocks = \
        set([u'emerald_block', u'diamond_block', u'redstone_block'])
    
    for i, block in enumerate(grid):
        if block in legal_blocks:
            start_states.append(0)
            legal_states.append(i)
        if block not in legal_blocks:
            start_states.append(3)
    start_states[end] = 2
    start_states[start] = 1
    start_states = np.array(start_states).reshape(21, 21)
    start_states = start_states[10:15, 7:12]
    # print(start_states)

    study = 1
    n_episode = 30
    env = Env(start, end)
    for i_episode in range(n_episode):
        # print(i_episode,'epoch')
        maze_img = env.start_env(start_states)
        # print(start_states)
        feat_maps = trans_torch(maze_img)
        loss = 0
        state = start
        EPSILON = 1 - 1 / (i_episode + 2)
        while True:
            # env.display()
            a = dqn.choose_action(feat_maps)  # 选择动作
            # 选动作, 得到环境反馈
            done, r, maze_img, state = env.step(a, state)
            next_feat_maps = trans_torch(maze_img)
            # 存记忆
            dqn.store_transition(feat_maps, a, r, next_feat_maps)
            if dqn.memory_counter > MEMORY_CAPACITY:
                if study == 1:
                    # log.info('2000经验池')
                    print(f"[epoch {i_episode+1}/{n_episode}] Experiences buffer full.")
                    study = 0
                loss = dqn.learn()  # 记忆库满了就进行学习
            if done == 1 or done == 2:  # 如果回合结束, 进入下回合
                # print("loss:",loss)
                # if done==1:
                # print('epoch',i_episode,r,'失败')

                if done == 2:
                    # log.info('epoch', i_episode, r, '成功')
                    print(f"[epoch {i_episode+1}/{n_episode}] Mission accomplished.")
                    break
            feat_maps = next_feat_maps

    path = get_shortest_path(dqn, mission_id+1, display=True)
    print(path)
    action_list = extract_action_list_from_path(path)

    print("Output (start,end)", (mission_id + 1), ":", (start, end))
    print("Output (path length)", (mission_id + 1), ":", len(path))
    print("Output (actions)", (mission_id + 1), ":", action_list)
    
    # Loop until mission ends:
    # action_index = 0
    
    # while world_state.is_mission_running:
        # # sys.stdout.write(".")
        # time.sleep(0.1)

        # # Sending the next commend from the action list -- found using the Dijkstra algo.
        # if action_index >= len(action_list):
            # print("Error:", "out of actions, but mission has not ended!")
            # time.sleep(2)
        # else:
            # agent_host.sendCommand(action_list[action_index])
        # action_index += 1
        # if len(action_list) == action_index:
            # # Need to wait few seconds to let the world state realise I'm in end block.
            # # Another option could be just to add no move actions -- I thought sleep is more elegant.
            # time.sleep(2)
        # world_state = agent_host.getWorldState()
        # for error in world_state.errors:
            # print("Error:", error.text)

    print()
    print("Mission", (mission_id + 1), "ended")

    
    # Mission has ended.
