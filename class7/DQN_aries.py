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
try:
    from malmo import MalmoPython
except:
    import MalmoPython
import os
import sys
import time
import json
import numpy as np
import random
from priority_dict import priorityDictionary as PQ

#参数
BATCH_SIZE = 32
LR = 0.01                   # 学习率
GAMMA = 0.9                 # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
TARGET_REPLACE_ITER = 5   # Q 现实网络的更新频率5次循环更新一次
MEMORY_CAPACITY = 2000     # 记忆库大小
N_ACTIONS = 4  # 棋子的动作0，1，2，3
N_STATES = 1

# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

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
                  <ServerQuitFromTimeUp timeLimitMs="10000000"/>
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


def load_grid(world_state):
    """
    Used the agent observation API to get a 21 X 21 grid box around the agent (the agent is in the middle).

    Args
        world_state:    <object>    current agent world state

    Returns
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)
    """
    while world_state.is_mission_running:
        # sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floorAll', 0)
            break
    return grid


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
    

def extract_action_list_from_path(path_list):
    """
    Converts a block idx path to action list.

    Args
        path_list:  <list>  list of block idx from source block to dest block.

    Returns
        action_list: <list> list of string discrete action commands (e.g. ['movesouth 1', 'movewest 1', ...]
    """
    action_trans = {-21: 'movenorth 1', 21: 'movesouth 1', -1: 'movewest 1', 1: 'moveeast 1',0:'meidong'}
    alist = []
    for i in range(len(path_list) - 1):
        curr_block, next_block = path_list[i:(i + 2)]
        alist.append(action_trans[next_block - curr_block])

    return alist


def get_shortest_path():
    # -------------------------------------
    # 根据最终得到的收敛后的DQN输出最优的路径
    # -------------------------------------
    s = env.start_env(map)
    s = trans_torch(s)
    s_path = []
    s_path.append(start)
    state = start
    while True:
        action = dqn.choose_action(s)
        done, r, s_ = env.step(action)
        s_ = trans_torch(s_)
        next_state = (env.x1+10)*21 + (env.y1+7)
        s_path.append(next_state)
        s = s_
        if done == 1:
            break
    return s_path


#变成三通道
def trans_torch(list1):
    list1=np.array(list1)
    l1=np.where(list1==1,1,0)
    l2=np.where(list1==2,1,0)
    l3=np.where(list1==3,1,0)
    b=np.array([l1,l2,l3])
    return b
#神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(3, 25, 5, 1, 0)
        self.f1 = nn.Linear(25, 16)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2 = nn.Linear(16, 4)
        self.f2.weight.data.normal_(0, 0.1)
    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.f1(x)
        x = F.relu(x)
        action = self.f2(x)
        return action
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net() #DQN需要使用两个神经网络
        #eval为Q估计神经网络 target为Q现实神经网络
        self.learn_step_counter = 0 # 用于 target 更新计时，100次更新一次
        self.memory_counter = 0 # 记忆库记数
        self.memory = list(np.zeros((MEMORY_CAPACITY, 4))) # 初始化记忆库用numpy生成一个(2000,4)大小的全0矩阵，
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample,x为场景
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x) #将场景输入Q估计神经网络
            #torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
            action = torch.max(actions_value, 1)[1].data.numpy() # 返回动作最大值
        else:   # 选随机动作
            action = np.array([np.random.randint(0, N_ACTIONS)]) # 比如np.random.randint(0,2)是选择1或0
        return action
    def store_transition(self, s, a, r, s_):
        # 如果记忆库满了, 就覆盖老数据，2000次覆盖一次
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = [s,a,r,s_]
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
        b_s=[]
        b_a=[]
        b_r=[]
        b_s_=[]
        for i in sample_index:
            b_s.append(self.memory[i][0])
            b_a.append(np.array(self.memory[i][1],dtype=np.int32))
            b_r.append(np.array([self.memory[i][2]],dtype=np.int32))
            b_s_.append(self.memory[i][3])
        b_s = torch.FloatTensor(b_s)#取出s
        b_a = torch.LongTensor(b_a) #取出a
        b_r = torch.FloatTensor(b_r) #取出r
        b_s_ = torch.FloatTensor(b_s_) #取出s_
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach Q现实
        q_target = b_r + GAMMA * q_next.max(1)[0].reshape(BATCH_SIZE, 1)   # shape (batch, 1) DQL核心公式
        loss = self.loss_func(q_eval, q_target) #计算误差
        # 计算, 更新 eval net
        self.optimizer.zero_grad() #
        loss.backward() #反向传递
        self.optimizer.step()
        return loss


#定义迷宫环境类
class Env():
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.start = start
        self.end = end
    def start_env(self, map):
        self.migong = map.copy() # 为简化计算可以仅获取迷宫中agent可站立的states
        self.x1 = self.start % 21 - 10
        self.y1 = self.start - (self.x1 + 10) * 21 - 7
        self.end_game = 0
        return  self.migong
    # 状态转移函数，输入动作，返回是否到达终点，本次动作的reward，本次动作更新后的迷宫地图
    def step(self, action):
        if action == 0:
            if self.x1 == 0:
                return (self.end_game,-1,self.migong)
            else:
                if self.migong[self.x1-1][self.y1] == 0:
                    self.migong[self.x1][self.y1] = 0
                    self.migong[self.x1-1][self.y1] = 1
                    self.x1 -=1
                    return (self.end_game,0,self.migong)
                elif self.migong[self.x1-1][self.y1] == 2:
                    self.end_game = 1
                    self.migong[self.x1][self.y1] = 0
                    self.migong[self.x1 - 1][self.y1] = 1
                    self.x1 -=1
                    return (self.end_game,100,self.migong)
                else:
                    self.end_game = 2
                    return (self.end_game,-1,self.migong)
        if action == 1:
            if self.x1 == 4:
                return (self.end_game,-1,self.migong)
            else:
                if self.migong[self.x1 + 1][self.y1] == 0:
                    self.migong[self.x1][self.y1] = 0
                    self.migong[self.x1 + 1][self.y1] = 1
                    self.x1 +=1
                    return (self.end_game, 0, self.migong)
                elif self.migong[self.x1 + 1][self.y1] == 2:
                    self.end_game = 1
                    self.migong[self.x1][self.y1] = 0
                    self.migong[self.x1 + 1][self.y1] = 1
                    self.x1 +=1
                    return (self.end_game, 100, self.migong)
                else:
                    self.end_game = 2
                    return (self.end_game, -1, self.migong)
        if action == 2:
            if self.y1 == 0:
                return (self.end_game,-1,self.migong)
            else:
                if self.migong[self.x1][self.y1-1] == 0:
                    self.migong[self.x1][self.y1] = 0
                    self.migong[self.x1][self.y1-1] = 1
                    self.y1 -=1
                    return (self.end_game,0,self.migong)
                elif self.migong[self.x1][self.y1-1] == 2:
                    self.end_game = 1
                    self.migong[self.x1][self.y1] = 0
                    self.migong[self.x1][self.y1-1] = 1
                    self.y1 -=1
                    return (self.end_game, 100, self.migong)
                else:
                    self.end_game = 2
                    return (self.end_game, -1, self.migong)
        if action == 3:
            if self.y1 == 4:
                return (self.end_game,-1,self.migong)
            else:
                if self.migong[self.x1][self.y1 + 1] == 0:
                    self.migong[self.x1][self.y1] = 0
                    self.migong[self.x1][self.y1+1] = 1
                    self.y1 +=1
                    return (self.end_game,0,self.migong)
                elif self.migong[self.x1][self.y1+1] == 2:
                    self.end_game = 1
                    self.migong[self.x1][self.y1] = 0
                    self.migong[self.x1][self.y1+1] = 1
                    self.y1 +=1
                    return (self.end_game, 100, self.migong)
                else:
                    self.end_game = 2
                    return (self.end_game, -1, self.migong)




#DQN实例化，由于5个任务迷宫地图基本相似，将DQN设在全局，后4次任务借鉴第1次结果可以更快收敛
dqn = DQN()

# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 5

for i in range(num_repeats):
    size = int(5)
    print("Size of maze:", size)
    my_mission = MalmoPython.MissionSpec(GetMissionXML("0", 0.4 + float(i / 20.0), size), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)
    # Attempt to start a mission:
    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "%s-%d" % ('Moshe', i))
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission", (i + 1), ":", e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission", (i + 1), "to start ", )
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        # sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission", (i + 1), "running.")

    grid = load_grid(world_state)
    start, end = find_start_end(grid)
    actions = []  # 定义actions
    legal_states = ['emerald_block','diamond_block','redstone_block']
    counter = 0
    map = []
    rewards = []
    for state in grid:
        if state in legal_states:
            map.append(0)
        else:
            map.append(3)
        counter += 1
    map[start] = 1
    map[end] = 2
    map = np.array(map).reshape(21,21)
    map = map[10:15,7:12]

    study = 1
    env = Env()

    #进行200次迭代，使DQN收敛
    for i_episode in range(1,201):
            EPSILON = 1 - 1 / (i_episode + 2)
            #当迭代次数大于100时，将EPSILON置为1.01，动作选取采取贪婪策略，不再有随机性，观察是否收敛
            if i_episode >= 100:
                EPSILON = 1.01
            s = env.start_env(map)
            s = trans_torch(s)
            loss = 0
            total_reward = 0
            num = 0
            while True:
                num += 1
                a = dqn.choose_action(s)
                done, r, s_  = env.step(a)
                total_reward += r
                s_ = trans_torch(s_)
                dqn.store_transition(s, a, r, s_)    
                if dqn.memory_counter > MEMORY_CAPACITY:
                    if study == 1:
                        print('2000 Experience Pool')
                        study = 0
                    loss = dqn.learn()  # 记忆库满了就进行学习
                if done == 1:  # 如果回合结束, 进入下回合
                    print(loss)
                    print('epoch', i_episode, r, 'success')
                    break
                s = s_
            average_reward = total_reward / num #计算每一次episode的平均回报
            if type(rewards) == np.ndarray:
                rewards = rewards.tolist()
            rewards.append(average_reward)
            rewards = np.array(rewards)
            if i_episode % 10 == 0:
                mean_episode_reward = np.mean(rewards[-10:])#计算10次episode的平均回报
                print('epiode',i_episode,'Evaluation Average Reward:',mean_episode_reward)#打印出10次episode的平均回报，判断是否收敛


    #EPSILON置为1.01，贪心策略选取最优路径
    EPSILON = 1.01
    path = get_shortest_path()
    action_list=extract_action_list_from_path(path)

    print("Output (start,end)", (i + 1), ":", (start, end))
    print("Output (path length)", (i + 1), ":", len(path))
    print("Output (actions)", (i + 1), ":", action_list)
    # Loop until mission ends:
    action_index = 0
    while world_state.is_mission_running:
        # sys.stdout.write(".")
        time.sleep(0.1)

        # Sending the next commend from the action list -- found using the Dijkstra algo.
        if action_index >= len(action_list):
            print("Error:", "out of actions, but mission has not ended!")
            time.sleep(2)
        else:
            agent_host.sendCommand(action_list[action_index])
        action_index += 1
        if len(action_list) == action_index:
            # Need to wait few seconds to let the world state realise I'm in end block.
            # Another option could be just to add no move actions -- I thought sleep is more elegant.
            time.sleep(2)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission", (i + 1), "ended")
    # Mission has ended.
