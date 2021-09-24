#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:58:27 2020

@author: godfp
"""


try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
from priority_dict import priorityDictionary as PQ
import numpy as np


actions = np.arange(4)
rewards = -1 * np.ones(441)
gamma = 0.9
# 设置迭代次数
n = 100000

GRID_SIZE = 441


#地图相关参数
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
                    <Seed>'''+str(seed)+'''</Seed>
                    <SizeAndPosition width="''' + str(size) + '''" length="''' + str(size) + '''" height="10" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
                    <StartBlock type="emerald_block" fixedToEdge="true"/>
                    <EndBlock type="redstone_block" fixedToEdge="true"/>
                    <PathBlock type="diamond_block"/>
                    <FloorBlock type="air"/>
                    <GapBlock type="air"/>
                    <GapProbability>'''+str(gp)+'''</GapProbability>
                    <AllowDiagonalMovement>false</AllowDiagonalMovement>
                  </MazeDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="10000"/>
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
            
#载入地图            
def load_grid(world_state):
    """
    Used the agent observation API to get a 21 X 21 grid box around the agent (the agent is in the middle).

    Args
        world_state:    <object>    current agent world state

    Returns
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)
    """
    while world_state.is_mission_running:
        #sys.stdout.write(".")
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

#找到起点、终点、air_block、diamond_block
def find_start_end(grid):
    """
    Finds the source and destination block indexes from the list.

    Args
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)

    Returns
        start: <int>   source block index in the list
        end:   <int>   destination block index in the list
    """
    #------------------------------------
    #
    #   Fill and submit this code
    #
#     return (None, None)
    #-------------------------------------
    counter = 0
    eb_index = None
    rb_index = None
    air_block=[]
    diamond_block=[]
    state=[]
    for i in grid:
        if i =='diamond_block':
            diamond_block.append(counter)
        
        if i =='air':
            air_block.append(counter)

        if i == 'emerald_block':
            eb_index = counter
           
        if i == 'redstone_block':
            rb_index = counter

        state.append(counter)    
        counter+=1
    
    return (eb_index, rb_index, air_block, diamond_block)

#从best_route获得agent的运动具体action（向东南西北）
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
        alist.append(action_trans[next_block - curr_block])

    return alist



# 定义状态动作转移，传入当前状态和执行的动作，返回当前状态下执行动作得到的转移概率、下一状态和奖励
def p_state_reward(state, action):
    # return((trans_prob,next_state,reward))

    
    # 向上移动
    if action == 0:
        #Fill and submit this code

        
    # 向下移动
    if action == 1:
         #Fill and submit this code

        
    # 向左移动
    if action == 2:
         #Fill and submit this code
        

        
    # 向右移动
    if action == 3:
         #Fill and submit this code

# 策略评估：计算策略下状态的价值
def compute_value_function(policy, gamma):
    ## 设置阈值
    theta = 0.5

    # 初始化每个状态的价值
    V = np.zeros(GRID_SIZE)

    # 创建每次迭代更新的状态价值表
    # ?
    
    ## 遍历所有状态
    for state in range(GRID_SIZE):
        ## 选择当前策略下当前状态所对应的动作
        for action in range(4):      # actions = range(4)
            tarns_prob, next_state, reward = \
                p_state_reward(state, action)
                 
        ## 返回当前状态下执行动作得到的转移概率、下一状态和奖励
        ## 计算策略下状态价值
    ## 价值表前后两次更新之差小于阈值时停止循环
    #Fill and submit this code
    return value_table


# 策略提升：更新策略
def next_best_policy(value_table, gamma):
    ## 创建空数组保存改进的策略
        ## 创建列表存储当前状态下执行不同动作的价值
        ## 遍历所有动作
            ## 返回当前状态-动作下一步的状态、转移概率和奖励
            ## 计算当前状态下执行当前动作的价值
        ## 策略提升：选取动作值最大的动作更新策略
    #Fill and submit this code
    return policy 


# 建立策略迭代函数
def policy_iteration(random_policy, gamma, n):
    policy = random_policy.copy()
    
    ## 进行迭代
    for _ in range(n):
        ## 策略评估：得到各状态的价值
        V = compute_value_function(policy, gamma)
    
        ## 策略提升：选取动作值最大的动作更新策略
        new_policy = next_best_policy(V, gamma)

        ## 对当前策略进行判断
        #?

        ## 替换为当前最佳策略
        policy = new_policy
    #Fill and submit this code
    return policy

# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 10

for i in range(num_repeats):
    states=[]
    size = int(6 + 0.5*i)
    print("Size of maze:", size)
    my_mission = MalmoPython.MissionSpec(GetMissionXML("0", 0.4 + float(i/20.0), size), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)
    # Attempt to start a mission:
    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "%s-%d" % ('Moshe', i) )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission", (i+1), ":",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission", (i+1), "to start ",)
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        #sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission", (i+1), "running.")

    grid = load_grid(world_state)
    s=['emerald_block','diamond_block','redstone_block']
    counter=0
    for i in grid:
        if i in s:
            states.append(counter)
        counter +=1
    print(counter)
    print(states)
    
    random_policy = 2 * np.ones(len(states))
    start,end,air_block,diamond_block=find_start_end(grid)
    print(start,end)
    
    best_policy = policy_iteration(random_policy, gamma, n)
    # 创建最佳路线列表，起始位置一定在状态start
    best_route = [start]
    next_state = start
    
    
    while True:
        
        # 通过最佳策略求解当前状态下执行最优动作所转移到的下一个状态
        _, next_state, _ = p_state_reward(next_state, best_policy[states.index(next_state)])
        # 将下个状态加入最佳路线列表
        best_route.append(next_state)
        
        # 转移到终止状态，停止循环
        if next_state == end:
            break
    
    print('best_route：',best_route)

    action_list = extract_action_list_from_path(best_route)
#     print("Output (start,end)", (i+1), ":", (start,end))
#     print("Output (path length)", (i+1), ":", len(best_route))
#     print("Output (actions)", (i+1), ":", action_list)
    # Loop until mission ends:
    action_index = 0
    while world_state.is_mission_running:
        #sys.stdout.write(".")
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
            print("Error:",error.text)

    print()
    print("ended")
    # Mission has ended.

