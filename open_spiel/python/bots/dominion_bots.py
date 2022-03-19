from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from open_spiel.python.games import dominion
from operator import itemgetter
import numpy as np
import pyspiel


class BigMoneyBot(pyspiel.Bot):
	def __init__(self, player_id, duchy_dancing = False):
		pyspiel.Bot.__init__(self)
		self._duchy_dancing = duchy_dancing
		self._player_id = player_id
	def step_with_policy(self,state: dominion.DominionGameState):
		legal_actions = state.legal_actions()
		if not legal_actions:
			return [], pyspiel.INVALID_ACTION
		obs_dict = state.observation_dict()
		turn_phase = obs_dict['TurnPhase'][0]
		if turn_phase is not dominion.TurnPhase.TREASURE_PHASE and turn_phase is not dominion.TurnPhase.BUY_PHASE:
			action = dominion.END_PHASE_ACTION
		elif turn_phase is dominion.TurnPhase.TREASURE_PHASE:
			unique_cards_in_hand = itemgetter(*np.nonzero(obs_dict['hand'])[0].tolist())(dominion._ALL_CARDS)
			if type(unique_cards_in_hand) is tuple:
				unique_cards_in_hand = list(unique_cards_in_hand)
			else:
				unique_cards_in_hand = [unique_cards_in_hand]
			action = next(filter(lambda card: isinstance(card,dominion.TreasureCard),unique_cards_in_hand)).play
		elif turn_phase is dominion.TurnPhase.BUY_PHASE:
			num_coins = obs_dict['coins'][0]
			if num_coins >= 3 and num_coins <= 5:
				action = dominion.SILVER.buy
			elif num_coins >= 6 and num_coins <= 7:
				action = dominion.GOLD.buy
			elif num_coins is 8:
				action = dominion.PROVINCE.buy
		policy = [(legal_action,1) if legal_action is action else (legal_action,0) for legal_action in legal_actions]
		return policy, action 
	def step(self,state):
		return self.step_with_policy(state)[1]
	def restart_at(self,state):
		pass