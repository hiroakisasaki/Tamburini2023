# -*- coding: utf-8 -*-
# Code by Fabio Tamburini, https://github.com/ftamburin

import sys
import numpy as np
import math
import random
import torch
from argparse import ArgumentParser

from EditDistanceWild import editdistance1N
from pycsa import CoupledAnnealer
from lsa_2g import lsa_2g


alphabets = {
				'uga':['m', 's', 'g', 'r', 'w', 'i', 'n', 'b', 'x', 'l', 'k', 'a', '$', 'y', 'q', '@', 'd', 'h', 'p', '<', '+', 'v', 'H', 't', '&', 'S', 'T', 'Z', 'z', 'u'],
				'heb':['m', 's', 'g', 'r', 'w', 'a', 'y', 'n', 'b', 'H', 'l', 'k', '$', 'q', 't', 'S', 'd', 'h', 'p', '<', 'z', '&', 'T'],
				'linb':['𐀀', '𐀁', '𐀂', '𐀃', '𐀄', '𐀅', '𐀆', '𐀇', '𐀈', '𐀉', '𐀊', '𐀋', '𐀍', '𐀏', '𐀐', '𐀑', '𐀒', '𐀓', '𐀔', '𐀕', '𐀖', '𐀗', '𐀘', '𐀙', '𐀚', '𐀛', '𐀜', '𐀝', '𐀞', '𐀟', '𐀠', '𐀡', '𐀢', '𐀣', '𐀤', '𐀥', '𐀦', '𐀨', '𐀩', '𐀪', '𐀫', '𐀬', '𐀭', '𐀮', '𐀯', '𐀰', '𐀱', '𐀲', '𐀳', '𐀴', '𐀵', '𐀶', '𐀷', '𐀸', '𐀹', '𐀺', '𐀼', '𐀽', '𐀿', '𐁀', '𐁁', '𐁂', '𐁄', '𐁅', '𐁆', '𐁇', '𐁈', '𐁉', '𐁊', '𐁋'],
				'greek':['α', 'ε', 'ι', 'ο', 'υ', 'δα', 'δε', 'δι', 'δο', 'δυ', 'ια', 'ιε', 'ιο', 'κα', 'κε', 'κι', 'κο', 'κυ', 'χα', 'χε', 'χι', 'χο', 'χυ', 'γα', 'γε', 'γι', 'γο', 'γυ', 'μα', 'με', 'μι', 'μο', 'μυ', 'να', 'νε', 'νι', 'νο', 'νυ', 'πα', 'πε', 'πι', 'πο', 'πυ', 'βα', 'βε', 'βι', 'βο', 'βυ', 'φα', 'φε', 'φι', 'φο', 'φυ', 'ρα', 'ρε', 'ρι', 'ρο', 'ρυ', 'λα', 'λε', 'λι', 'λο', 'λυ', 'σα', 'σε', 'σι', 'σο', 'συ', 'τα', 'τε', 'τι', 'το', 'τυ', 'θα', 'θε', 'θι', 'θο', 'θυ', 'fα', 'fε', 'fι', 'fο', 'ζα', 'ζε', 'ζο', 'hα', 'αι', 'αυ', 'δfε', 'δfο', 'νfα', 'πτε', 'ρια', 'ραι', 'ριο', 'τια', 'τfε', 'τfο'],
				'csyl':['𐠀', '𐠁', '𐠂', '𐠃', '𐠄', '𐠲', '𐠳', '𐠴', '𐠵', '𐠼', '𐠿', '𐠅', '𐠈', '𐠊', '𐠋', '𐠌', '𐠍', '𐠎', '𐠏', '𐠐', '𐠑', '𐠒', '𐠓', '𐠔', '𐠕', '𐠖', '𐠗', '𐠘', '𐠙', '𐠚', '𐠛', '𐠜', '𐠝', '𐠷', '𐠸', '𐠞', '𐠟', '𐠠', '𐠡', '𐠢', '𐠣', '𐠤', '𐠥', '𐠦', '𐠧', '𐠨', '𐠩', '𐠪', '𐠫', '𐠬', '𐠭', '𐠮', '𐠯', '𐠰', '𐠱']
			}



def ReadLexicon(fname, stream, sep):
	Lex = []
	groups = []
	maxl = 0
	with open(fname, "r", encoding='utf-8') as f:
		first = True
		for l in f:
			if (first):
				l = l.rstrip().split()
				if (l[stream] in alphabets):
					C = alphabets[l[stream]]
					print('SIGNS FOR ',l[stream],':',C, file=sys.stderr)
					first = False
				else:
					print('ERROR: file',fname,'must contain an alphabet label in the first line!')
					sys.exit(1)
			else:
				l = l.rstrip().split()[stream]
				if ((sep == '-') and (l != '_')):
					l = l.split('-')[1:-1]
					l = ''.join(l)
				if (l != '_'):
					l = l.split('|')
					ng = 0
					for ll in l:
						ll = list(ll)
						if (len(ll) > maxl):
							maxl = len(ll)
						Lex.append(ll)
						ng += 1
					groups.append(ng)
	return C, Lex, maxl, np.array(groups)



class Problem():
	def __init__(self, fName, fNameFix, sep, N, M, device): 
		self.lC, self.lLex, self.maxLl, _ = ReadLexicon(fName, 0, sep)
		self.kC, self.kLex, self.maxLk, self.kGroups = ReadLexicon(fName, 1, sep)

		self.fix = {}
		with open(fNameFix, "r", encoding='utf-8') as f:
			for l in f:
				l = l.rstrip().split()
				self.fix[l[0]] = l[1:]
		for fL,fK in self.fix.items():
			if (fL in self.lC):
				self.lC.remove(fL)
			for j in fK:
				if (j in self.kC):
					self.kC.remove(j)
		if (self.fix != {}):
			print('FIXED DICTIONARY:',self.fix, file=sys.stderr)
		print('|lLex|', len(self.lLex), self.maxLl, file=sys.stderr)
		print(self.lC, len(self.lC), file=sys.stderr)
		print('|kLex|', len(self.kLex), self.maxLk, file=sys.stderr)
		print(self.kC, len(self.kC), file=sys.stderr)
		self.llC = len(self.lC)
		self.lkC = len(self.kC)

		if (N*self.llC <= self.lkC):
			print('ERROR, wrong parameters:',N*self.llC,'MUST BE >',self.lkC, file=sys.stderr)
			sys.exit(1)
		assert (M == 1) or (M == 2)
		assert (N == 1) or (N == 2) or (N == 3)

		self.N = N
		self.M = M
		self.ED = (1,1)	# WEIGHTs FOR THE EDIT DISTANCE
		print('N =',N,'   M =',M, file=sys.stderr)
		print('ED =',self.ED, file=sys.stderr)
		self.device = device
		print('WORKING on',self.device, file=sys.stderr)

		# PREPARE THE SECOND LEXICON AND MOVE IT ON GPU (if required)
		K = []
		for i in range(len(self.kLex)):
			K += [([ord(c) for c in self.kLex[i]]+[0]*self.maxLk)[:self.maxLk]]
		self.y = torch.tensor(K, dtype=torch.long, requires_grad = False).to(self.device)

		# INITIAL STATE
		self.init_state = [i for i in range(N*self.llC)]
		random.shuffle(self.init_state)
		self.m = [0, N*self.llC-self.lkC, N*self.llC]
		if (M == 2):
			m2st = [i for i in range(N*self.llC)]
			random.shuffle(m2st)
			self.init_state = m2st + self.init_state
			self.m = self.m + [N*self.llC+x for x in self.m[1:]]
		print('INITIAL STATE',self.init_state, len(self.init_state), file=sys.stderr)
		print('STATE MARGINS',self.m, file=sys.stderr)


	def State2Assignment(self,X):
		Po = [-1]*self.llC
		for i in range(self.M):
			P = [x % self.llC for x in X[self.m[2*i+1]:self.m[2*i+2]]]
			for j in range(len(P)):
				jj = j % self.lkC
				if (Po[P[j]] != -1):
					if (jj not in Po[P[jj]]):
						Po[P[j]] = Po[P[j]]+[jj]
				else:
					Po[P[j]] = [jj]
		return Po


	def energy(self, state, freeMem=True):
		def expandWord(w, i, l):
			if (i == len(w)):
				yield l
			else:
				if (w[i] in self.fix):
					for j in range(len(w[i])):
						yield from expandWord(w, i+1, l+[ord(x) for x in self.fix[w[i]][j]])
				else:
					ci = self.lC.index(w[i])
					if (P[ci] != -1):
						for j in range(len(P[ci])):
							yield from expandWord(w, i+1, l+[ord(x) for x in self.kC[P[ci][j]]])
					else:
						yield from expandWord(w, i+1, l+[ord(x) for x in w[i]])

		P = self.State2Assignment(state)
		self.costM = []
		self.lGroups = []
		self.dynLex = []
		for j in range(len(self.lLex)):
			nel = 0
			for ww in expandWord(self.lLex[j], 0, []):
				self.dynLex += [self.lLex[j]]
				L = [(ww+[0]*self.maxLl)[:self.maxLl]]
				x = torch.tensor(L, dtype=torch.long, requires_grad = False).to(self.device)
				pred = editdistance1N(x, self.y, 0, self.ED[0], self.ED[1]).to('cpu')
				self.costM.append(pred[:,1].numpy())
				del pred
				nel += 1
			self.lGroups.append(nel)
		self.costM = np.array(self.costM)
		assert len(self.lLex) == len(self.lGroups)
		self.lGroups = np.array(self.lGroups)
		self.row_ind, self.col_ind, out = lsa_2g(self.costM, self.lGroups, self.kGroups)

		# PENALISATIONS...
		lPN = 0
		lPM = 0
		penF = 4.0
		if (self.M > 1):
			# ...FOR MULTIPLE ASSIGNMENTS
			lPM = sum([1 if ((lP!=-1) and (len(lP)>1)) else 0 for lP in P])
			out = out + penF*lPM
		# ...FOR LOST SIGNS UNASSIGNED
		lPN = sum([1 if (lP==-1) else 0 for lP in P])
		out = out + penF*lPN
		if (freeMem):
			del self.costM, self.lGroups
		print(out, state, P, lPN, lPM, self.ED, 'EnEval')
		sys.stdout.flush()
		return out

	def move(self, state, tgen, qa=False):
		if (qa):	
			qaf = max(int(len(state) * tgen), 1)
		else:
			qaf = 1
		for j in range(qaf):
			if (self.M == 1):
				a = random.randint(self.m[0], self.m[2]-1)
				b = random.randint(self.m[1], self.m[2]-1)
			else:
				x = random.randint(self.m[0], self.m[4]-1)
				if (x < self.m[2]):
					a = random.randint(self.m[0], self.m[2]-1)
					b = random.randint(self.m[1], self.m[2]-1)
				else:
					a = random.randint(self.m[2], self.m[4]-1)
					b = random.randint(self.m[3], self.m[4]-1)
			state[a], state[b] = state[b], state[a]
		return state
		

def EvalModel(UsolMatch, goldMatches):
	# COMPUTE Luo et al. 2019 measure
	print('COMPUTING LUO et al. 2019 Measure')
	found = 0
	for wL,wsK in goldMatches:
		wsK = wsK.split('|')
		match = False
		for wK in wsK:
			if ((wL,wK) in UsolMatch):
				match = True
				wmatch = wK
		if (match):
			found += 1
			print((wL,wmatch),wsK,'\tOK!')
		else:
			for a,b in UsolMatch:
				if (a == wL):
					print((wL,b),wsK,'\t.')
	luo = found / len(goldMatches)
	print('Accuracy:',luo,str(found)+'/'+str(len(goldMatches)))



if __name__ == '__main__':
	argument_parser = ArgumentParser()
	argument_parser.add_argument('-c', "--cog-file", required=True, help="File containing cognates.")
	argument_parser.add_argument("-f", "--fix-file", required=True, help="File containing the fixed signs.")
	argument_parser.add_argument("-s", "--sep", default="", help="The separator used in COG file.")
	argument_parser.add_argument("-n", "--n", required=True, type=int, help="N value for solution design.")
	argument_parser.add_argument("-m", "--m", required=True, type=int, help="M value for solution design.")
	argument_parser.add_argument("-d", "--device", default="cpu", help="Select cpu or cuda device.")
	argument_parser.add_argument("-o", "--sol", default="", help="Evaluate considering this solution.")
	args = argument_parser.parse_args()


	prob = Problem(args.cog_file, args.fix_file, args.sep, args.n, args.m, args.device)

	if (args.sol == ""):
		# TRAINING
		# Initialize the CSA process.
		n_annealers = 16
		processes = 8
		steps = 100000
		stepsatsameT = max(math.ceil(len(prob.init_state) / n_annealers), 5)
		annealer = CoupledAnnealer(
			prob.energy,
			prob.move,
			initial_state=[prob.init_state] * n_annealers,
			update_interval=stepsatsameT,	
			steps=steps,
			processes=processes,
			n_annealers=n_annealers,
			tacc_initial=200.0,
			tacc_schedule=0.95,	
			tgen_initial=1.0,	
			tgen_schedule=0.999,	
			qa=0.1,
			verbose=1,
			device=prob.device)

		print('STARTING ANNEALING:', n_annealers, processes, steps, stepsatsameT, file=sys.stderr)
		sys.stderr.flush()
		
		annealer.anneal()

		# Get the best result from all `n_annealers`.
		energy, state = annealer.get_best()

		P = prob.State2Assignment(state)
		print("\n\nEnergy",energy, file=sys.stderr)
		print("State",state, file=sys.stderr)
		print("Assignment",P, file=sys.stderr)
		for j in range(len(P)):
			print(prob.lC[j], file=sys.stderr, end=' ')
			if (type(P[j]) == list):
				for i in range(len(P[j])):
					print(prob.kC[P[j][i]], file=sys.stderr, end=' ')
			print(file=sys.stderr)
	else:
		# CHECK ONE SOLUTION
		exec('state = '+args.sol)
		P = prob.State2Assignment(state)
		print('State =',state,P)
		for j in range(len(P)):
			print(prob.lC[j], file=sys.stderr, end=' ')
			if (type(P[j]) == list):
				for i in range(len(P[j])):
					print(prob.kC[P[j][i]], file=sys.stderr, end=' ')
			print(file=sys.stderr)
		print('--------', file=sys.stderr)
		sys.stderr.flush()
		prob.energy(state, freeMem=False)

		# MATCHING LEXICA
		row_ind, col_ind, out = lsa_2g(prob.costM, prob.lGroups, prob.kGroups)
		UsolMatch = []
		for l in range(len(row_ind)):
			UsolMatch.append((''.join(prob.dynLex[row_ind[l]]),''.join(prob.kLex[col_ind[l]])))

		goldMatches = []
		with open(args.cog_file, "r", encoding='utf-8') as f:
			first = True
			for l in f:
				l = l.rstrip().split()
				if (first):
					lLabel, kLabel = l[0], l[1]
					first = False
				else:	
					if ((l[0] != '_') and (l[1] != '_')):
						goldMatches.append((l[0], l[1]))

		EvalModel(UsolMatch, goldMatches)
