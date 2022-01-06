# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyspiel

_MIN_PLAYERS = 2
_MAX_PLAYERS = 4


_GAME_TYPE = pyspiel.GameType(
 short_name="python Dominion",
 long_name = "python Dominion",
 dynamics = pyspiel.GameType.Dynamics.SEQUENTIAL,
 chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
 information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
 utility=pyspiel.GameType.Utility.ZERO_SUM,
 reward_model=pyspiel.GameType.RewardModel.TERMINAL,
 max_num_players=_MAX_PLAYERS,
 min_num_players= _MIN_PLAYERS,
 provides_information_state_string=True,
 provides_information_state_tensor=True,
 provides_observation_string=True,
 provides_observation_tensor=True,
 provides_factored_observation_string=True
)


class Card(object):
 """
 Base class for all cards.
 """
 def __init__(self, name: str, cost: int):
  self.name = name
  self.cost = cost

class VictoryCard(Card):
 def __init__(self, name: str, cost: int, victory_points: int, vp_fn: callable):
  super(Victorycard,self).__init__(name,cost)
  self.victory_points = victory_points
 
class TreasureCard(Card):
 def __init__(self, name: str, cost: int, coins: int):
  super(TreasureCard,self).__init__(name,cost)
  self.coins = coins

class ActionCard(Card):
 def __init__(self, name: str, cost: int, add_actions: int = 0, add_buys: int = 0,add_cards: int = 0 , is_attack: bool = False,coins: int = 0, effect_fn: callable = None, effect_list: list = []):
  super(ActionCard,self).__init__(name,cost)
  self.add_actions = add_actions
  self.add_buys = add_buys
  self.add_cards = add_cards
  self.is_attack = is_attack
  self.effect_list = effect_list
  self.effect_fn = effect_fn
  self.coins = coins


class ReactionCard(Card):
 def __init__(name: str, cost: int, add_cards: int, global_trigger: callable): 
  super(ReactionCard,self).__init__(name,cost)
  self.add_cards = add_cards
  self.global_trigger = global_trigger

class AttackCard(ActionCard):
  def __init__(name: str, cost: int, add_cards: int, global_trigger: callable): 
    super(AttackCard,self).__init__(name,cost,add_cards=add_cards)
    self.global_trigger = global_trigger
    
class ReactionCard(ActionCard)
  def __init__(name: str, cost: int, add_cards: int, global_trigger: callable): 
    super(AttackCard,self).__init__(name,cost,add_cards=add_cards)
    self.global_trigger = global_trigger
  
_BASE_KINGDOM_CARDS = [
  ActionCard(name='Village',cost=3,add_actions=2,add_cards=1),
  ActionCard(name='Laboratory',cost=5,add_cards=2,add_actions=1),
  ActionCard(name='Market',cost=5,add_actions=1,add_buys=1,coins=1,add_cards=1),
  ActionCard(name='Festival',cost=5, add_actions=2,add_buys=1,coins=2),
  ActionCard(name="Smithy",cost=4,add_cards=3),
  AttackCard(name='Militia',cost=4,coins=2,effect_list=[]]),
  VictoryCard(name='Gardens',cost=4,vp_fn = all_cards: math.floor(len(all_cards)/10)),
  ActionCard(name='Chapel',cost=2,effect_list=[]),
  AttackCard(name='Witch',cost=5,add_cards=2,effect_list=[]),
  ActionCard(name='Workshop',cost=3,effect_list=[],effect_list=[]),
  AttackCard(name='Bandit',cost=5,effect_list=[],effect_fn=None),
  ActionCard(name='Remodel',cost=4,effect_list=[]),
  ActionCard(name='Throne Room',cost=4,effect_fn=None),
  ActionCard(name='Moneylender',cost=4,effect_fn=None),
  ActionCard(name='Poacher',cost=4,add_cards=4,add_actions=1,coins=1,effect_fn=None),
  ActionCard(name='Merchant',cost=3,add_cards=1,add_actions=1,effect_fn=None),
  ActionCard(name='Cellar',cost=2,effect_fn=None),
  ActionCard(name='Mine',cost=5,effect_fn=None),
  ActionCard(name='Vassal',cost=3,coins=2,effect_fn=None),
  ActionCard(name='Council Room',cost=5,add_cards=4,add_buys=1,effect_fn=None),
  ActionCard(name='Artisan',cost=6,effect_fn=None),
  ActionCard(name='Bureaucrat',cost=4,effect_fn=None),
  ActionCard(name='Sentry',cost=5,add_cards=1,add_actions=1,effect_fn=None),
  ActionCard(name='Harbinger',cost=3,add_cards=1,add_actions=1,effect_list=[]),
  ActionCard(name='Library',cost=5,effect_fn=None),
  ReactionCard(name='Moat',cost=2,add_cards=2,global_trigger=None)
]

_GAME_INFO = pyspiel.GameInfo(
 num_distinct_actions = len(_BASE_KINGDOM_CARDS),
 max_chance_outcomes=0,
 num_players=2,
 min_utility=-1.0,
 max_utility=1.0,
 utility_sum=0,
 max_game_length=3
)
