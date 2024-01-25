#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
#################################
# Generate PU and SU trajectory #
#################################

################################
# (-20,-30,1)   (20,-30,1)
#       ____ ____ 
#      |         | 
#      |         |
#      |         | 
#      |____ ____|
#
# (-20,-70,1)   (20,-70,1)
################################

class UT:
	def __init__(self, InitX, InitY, TargetX, TargetY):
		self.CurX = InitX
		self.CurY = InitY
		self.TargetX = TargetX
		self.TargetY = TargetY
		self.Trajectory = []
		self.State = False # True means this user has arrived the destination

	def saveTrajectory(self):
		self.Trajectory.append([self.CurX, self.CurY, 1])

	def move(self):
		if abs(self.TargetX-self.CurX)>=abs(self.TargetY-self.CurY):
			moveX = lambda Cx, Tx: (Cx - 1) if Cx > Tx else Cx + 1
			self.CurX = moveX(self.CurX, self.TargetX)
		else:
			moveY = lambda Cy, Ty: (Cy - 1) if Cy > Ty else Cy + 1
			self.CurY = moveY(self.CurY, self.TargetY)

		if (self.TargetX==self.CurX)&(self.TargetY==self.CurY):
			self.State = True

def TrainData():
	Location = [[-20, -30, 20, -30], [-20, -30, -20, -70], [-20, -70, 20, -70], [20, -30, 20, -70], [0, -30, 0, -70]]

	for i in range(len(Location)):
		Agent = UT(Location[i][0], Location[i][1], Location[i][2], Location[i][3])
		Agent.saveTrajectory()
		while True:
			Agent.move()
			Agent.saveTrajectory()
			if Agent.State == True:
				break

		Traj_data = np.array(Agent.Trajectory)
		Traj_data.reshape(3, len(Traj_data))
		np.savetxt("Train_Trajectory_User_MDP"+str(i)+".csv", Traj_data, delimiter=',')	

def TestData():

	Location = [[-20, -30, 20, -30], [-20, -30, -20, -70], [-20, -70, 20, -70], [20, -30, 20, -70], [0, -30, 0, -70]]

	for i in range(len(Location)):
		Agent = UT(Location[i][0], Location[i][1], Location[i][2], Location[i][3])
		Agent.saveTrajectory()
		while True:
			Agent.move()
			Agent.saveTrajectory()
			if Agent.State == True:
				break
		Traj_data = np.array(Agent.Trajectory)
		Traj_data.reshape(3, len(Traj_data))
		np.savetxt("Test_Trajectory_User_MDP"+str(i)+".csv", Traj_data, delimiter=',')

if __name__ == '__main__':
	TrainData()
	TestData()
