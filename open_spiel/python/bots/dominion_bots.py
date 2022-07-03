from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import rand
from open_spiel.python.games import dominion
from operator import itemgetter
import random 
import numpy as np
import pyspiel


class BigMoneyBot(pyspiel.Bot):
	def __init__(self, observer, player_id, duchy_dancing = False):
		pyspiel.Bot.__init__(self)
		self.observer = observer
		self._duchy_dancing = duchy_dancing
		self._player_id = player_id
	
	def player_id(self):
		return self._player_id
	
	def provides_policy(self):
		return True
		
	@staticmethod
	def purchase_treasure_card_if_avail(card: dominion.TreasureCard, state):
		return card.buy if state.supply_piles[card.name].qty > 0 else dominion.END_PHASE_ACTION
	
	def random_policy(self,legal_actions):
		p = 1 / len(legal_actions)
		action = random.choice(legal_actions)
		policy = [(action, p) for action in legal_actions]
		return policy, action

	def _is_penultimate_province(self):
		return self.observer.dict['victory_piles'].tolist()[3] == 1
	
	def _duchy_available(self):
		return self.observer.dict['victory_piles'].tolist()[2] > 0

	def play_first_treasure_card(self):
		obs_dict = self.observer.dict
		unique_cards_in_hand = itemgetter(*np.nonzero(obs_dict['hand'])[0].tolist())(dominion._ALL_CARDS)
		if isinstance(unique_cards_in_hand,tuple):
			return next(filter(lambda card: isinstance(card, dominion.TreasureCard),unique_cards_in_hand)).play
		else:
			return unique_cards_in_hand.play
		
	def step_with_policy(self,state):
		legal_actions = state._legal_actions(self._player_id)

		if legal_actions == [dominion.END_PHASE_ACTION]:
			policy = [(dominion.END_PHASE_ACTION,1)]
			return policy, dominion.END_PHASE_ACTION
		if not legal_actions:
			return [], pyspiel.INVALID_ACTION
		
		self.observer.set_from(state,self.player_id())
		obs_dict = self.observer.dict
		turn_phase = dominion.TurnPhase(obs_dict['TurnPhase'].tolist()[0])
		effect = obs_dict['effect'].tolist()[0]

		if effect != 0:
			return self.random_policy(legal_actions)

		action = dominion.END_PHASE_ACTION
		if turn_phase is dominion.TurnPhase.TREASURE_PHASE:
			action = self.play_first_treasure_card()
		elif turn_phase is dominion.TurnPhase.BUY_PHASE:
			num_coins = obs_dict['coins'].tolist()[0]
			if num_coins >= 3 and num_coins <= 5:
				action = BigMoneyBot.purchase_treasure_card_if_avail(dominion.SILVER,state)
			elif num_coins >= 6 and num_coins <= 7:
				action = BigMoneyBot.purchase_treasure_card_if_avail(dominion.GOLD,state)
			elif num_coins >= 8:
				if self._duchy_dancing and self._is_penultimate_province() and self._duchy_available():
					action = dominion.DUCHY.buy
				else:
					action = dominion.PROVINCE.buy
		else:
			return self.random_policy(legal_actions)
		#deterministic policy
		policy = [(legal_action,1) if legal_action is action else (legal_action,0) for legal_action in legal_actions]
		return policy, action 
	def step(self,state):
		return self.step_with_policy(state)[1]
	def restart_at(self,state):
		pass
