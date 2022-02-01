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

"""
Hands are hidden 
"""

import pyspiel
import random
from operator import itemgetter 

_MIN_PLAYERS = 2
_MAX_PLAYERS = 4
_HAND_SIZE = 5
_DRAW_PILE_SIZE = 10

""" Default kingdom cards to be used for the game. """
_PRESET_KINGDOM_CARDS = ['Village', 'Bureaucrat', 'Smithy', 'Witch', 'Militia', 'Moat', 'Library', 'Market', 'Mine',
                         'Council Room']

_DEFAULT_PARAMS = {
    'num_players': _MIN_PLAYERS,
    'automate_action_phase': True
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python Dominion",
    long_name="python Dominion",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.CONSTANT_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_MAX_PLAYERS,
    min_num_players=_MIN_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
    default_loadable=True,
    parameter_specification=_DEFAULT_PARAMS)


class Card(object):
    """
    Base class for all cards.
    """

    def __init__(self, name: str, cost: int):
        self.name = name
        self.cost = cost


class VictoryCard(Card):
    def __init__(self, name: str, cost: int, victory_points: int, vp_fn: callable = None):
        super(VictoryCard, self).__init__(name, cost)
        self.victory_points = victory_points


class TreasureCard(Card):
    def __init__(self, name: str, cost: int, coins: int):
        super(TreasureCard, self).__init__(name, cost)
        self.coins = coins


class ActionCard(Card):
    def __init__(self, name: str, cost: int, add_actions: int = 0, add_buys: int = 0, add_cards: int = 0,
                 is_attack: bool = False, coins: int = 0, effect_fn: callable = None, effect_list: list = []):
        super(ActionCard, self).__init__(name, cost)
        self.add_actions = add_actions
        self.add_buys = add_buys
        self.add_cards = add_cards
        self.is_attack = is_attack
        self.effect_list = effect_list
        self.effect_fn = effect_fn
        self.coins = coins


class AttackCard(ActionCard):
    def __init__(name: str, cost: int, add_cards: int, global_trigger: callable):
        super(AttackCard, self).__init__(name=name, cost=cost, add_cards=add_cards)
        self.global_trigger = global_trigger


class ReactionCard(ActionCard):
    def __init__(name: str, cost: int, add_cards: int, global_trigger: callable):
        super(ReactionCard, self).__init__(name, cost, add_cards=add_cards)
        self.global_trigger = global_trigger


class Pile(object):
    def __init__(self, card, qty):
        self.card = card
        self.qty = qty

    def __str__(self):
        return str(self.card)


class SupplyPile(Pile):
    def __init__(self, card, qty, buyable=True):
        super(SupplyPile, self).__init__(card, qty)
        self.buyable = buyable


VILLAGE = ActionCard(name='Village', cost=3, add_actions=2, add_cards=1),
LABORATORY = ActionCard(name='Laboratory', cost=5, add_cards=2, add_actions=1),
MARKET = ActionCard(name='Market', cost=5, add_actions=1, add_buys=1, coins=1, add_cards=1),
FESTIVAL = ActionCard(name='Festival', cost=5, add_actions=2, add_buys=1, coins=2),
SMITHY = ActionCard(name="Smithy", cost=4, add_cards=3),
# MILITIA = AttackCard(name='Militia',cost=4,coins=2,effect_list=[]),
GARDENS = VictoryCard(name='Gardens', cost=4, victory_points=0,
                      vp_fn=lambda all_cards: math.floor(len(all_cards) / 10)),
CHAPEL = ActionCard(name='Chapel', cost=2, effect_list=[]),
# WITCH = AttackCard(name='Witch',cost=5,add_cards=2,effect_list=[]),
WORKSHOP = ActionCard(name='Workshop', cost=3, effect_list=[]),
# BANDIT = AttackCard(name='Bandit',cost=5,effect_list=[],effect_fn=None),
REMODEL = ActionCard(name='Remodel', cost=4, effect_list=[]),
THRONE_ROOM = ActionCard(name='Throne Room', cost=4, effect_fn=None),
MONEYLENDER = ActionCard(name='Moneylender', cost=4, effect_fn=None),
POACHER = ActionCard(name='Poacher', cost=4, add_cards=4, add_actions=1, coins=1, effect_fn=None),
MERCHANT = ActionCard(name='Merchant', cost=3, add_cards=1, add_actions=1, effect_fn=None),
CELLAR = ActionCard(name='Cellar', cost=2, effect_fn=None),
MINE = ActionCard(name='Mine', cost=5, effect_fn=None),
VASSAL = ActionCard(name='Vassal', cost=3, coins=2, effect_fn=None),
COUNCIL_ROOM = ActionCard(name='Council Room', cost=5, add_cards=4, add_buys=1, effect_fn=None),
ARTISAN = ActionCard(name='Artisan', cost=6, effect_fn=None),
BUREAUCRAT = ActionCard(name='Bureaucrat', cost=4, effect_fn=None),
SENTRY = ActionCard(name='Sentry', cost=5, add_cards=1, add_actions=1, effect_fn=None),
HARBINGER = ActionCard(name='Harbinger', cost=3, add_cards=1, add_actions=1, effect_list=[]),
LIBRARY = ActionCard(name='Library', cost=5, effect_fn=None),
# MOAT = ReactionCard(name='Moat',cost=2,add_cards=2,global_trigger=None)


INIT_KINGDOM_SUPPLY = {
    "Village": SupplyPile(VILLAGE, 10),
    "Laboratory": SupplyPile(LABORATORY, 10),
    "Market": SupplyPile(MARKET, 10),
    "Festival": SupplyPile(FESTIVAL, 10),
    "Smithy": SupplyPile(SMITHY, 10),
    # "Militia": SupplyPile(MILITIA, 10),
    "Gardens": SupplyPile(GARDENS, 8),
    "Chapel": SupplyPile(CHAPEL, 10),
    # "Witch": SupplyPile(WITCH, 10),
    "Workshop": SupplyPile(WORKSHOP, 10),
    # "Bandit": SupplyPile(BANDIT, 10),
    "Remodel": SupplyPile(REMODEL, 10),
    "Throne Room": SupplyPile(THRONE_ROOM, 10),
    "Moneylender": SupplyPile(MONEYLENDER, 10),
    "Poacher": SupplyPile(POACHER, 10),
    "Merchant": SupplyPile(MERCHANT, 10),
    "Cellar": SupplyPile(CELLAR, 10),
    "Mine": SupplyPile(MINE, 10),
    "Vassal": SupplyPile(VASSAL, 10),
    "Council Room": SupplyPile(COUNCIL_ROOM, 10),
    "Artisan": SupplyPile(ARTISAN, 10),
    "Bureaucrat": SupplyPile(BUREAUCRAT, 10),
    "Sentry": SupplyPile(SENTRY, 10),
    "Harbinger": SupplyPile(HARBINGER, 10),
    "Library": SupplyPile(LIBRARY, 10),
    # "Moat": SupplyPile(MOAT, 10),
}

COPPER = TreasureCard(name='Copper', cost=0, coins=1)
SILVER = TreasureCard(name='Silver', cost=3, coins=2)
GOLD = TreasureCard(name='Gold', cost=6, coins=3)

_INIT_TREASURE_CARDS_SUPPLY = {'Copper': SupplyPile(COPPER, 46), 'Silver': SupplyPile(SILVER, 40),
                               'Gold': SupplyPile(GOLD, 30)}

CURSE = VictoryCard(name='Curse', cost=0, victory_points=-1),
DUCHY = VictoryCard(name='Duchy', cost=4, victory_points=5)
ESTATE = VictoryCard(name='Estate', cost=2, victory_points=1)
PROVINCE = VictoryCard(name='Province', cost=8, victory_points=8)

_INIT_VICTORY_CARDS_SUPPLY = {'Curse': SupplyPile(CURSE, 10), 'Estate': SupplyPile(ESTATE, 8),
                              'Duchy': SupplyPile(DUCHY, 8), 'Province': SupplyPile(PROVINCE, 8)}


class TurnPhase(enumerate):
    ACTION_PHASE = 1
    TREATURE_PHASE = 2
    BUY_PHASE = 3
    END_PHASE = 4

class Player(object):
    def __init__(self,id):
        self.id = id
        self.victory_points = 0 
        self.draw_pile = [COPPER for _ in range(7)] + [ESTATE for _ in range(3)]
        self.discard_pile = []
        self.trash_pile = []
        self.hand = []
        self.phase = TurnPhase.END_PHASE 
        self.actions = 0
        self.buys = 0
        self.coins = 0
    
    def _draw_hand(self): 
        """ draw a player's hand (5 cards) from their draw pile """
        cards_drawn_idxs = set(random.sample(range(len(self.draw_pile)),_HAND_SIZE-len(self.hand)))
        self.hand = itemgetter(*cards_drawn_idxs)(self.draw_pile)
        self.draw_pile = [card for idx,card in enumerate(self.draw_pile) if not idx in cards_drawn_idxs]
    
    def init_turn(self):
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.phase = TurnPhase.ACTION_PHASE
        self._draw_hand()


class DominionGame(pyspiel.Game):
    """ A python version of Dominion."""

    def __init__(self, params=None):
        _GAME_INFO = pyspiel.GameInfo(
            num_distinct_actions=len(_PRESET_KINGDOM_CARDS) + len(_INIT_TREASURE_CARDS_SUPPLY) + len(
                _INIT_VICTORY_CARDS_SUPPLY),
            max_chance_outcomes=0,
            num_players=params["num_players"],
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0,
            max_game_length=3
        )
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def get_game_info(self):
        return self._GAME_INFO

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return DominionGameState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        pass


class DominionGameState(pyspiel.State):
    """ a python version of the Dominon state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)

        self._players = [Player(id=i) for i in range(game.num_players())]

        # state
        self.draw_piles = [player.draw_pile for player in self._players]
        self.victory_points = [player.victory_points for player in self._players]
        self.hands = [player.hand for player in self._players]
        self.discard_piles = [player.discard_pile for player in self._players]
        self.trash_piles = [player.trash_pile for player in self._players]
        self.kingdom_piles = {key: INIT_KINGDOM_SUPPLY[key] for key in INIT_KINGDOM_SUPPLY if
                              key in _PRESET_KINGDOM_CARDS}
        self.treasure_piles = _INIT_TREASURE_CARDS_SUPPLY
        self.victory_piles = _INIT_VICTORY_CARDS_SUPPLY
        self._cur_player = 0
        self._is_terminal = False
        
        self._start_game()

    def _start_game(self):
        self._players[self._cur_player].init_turn()

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        """ depending on turn phase
        if TREASURE_PHASE: legal action is to 1. buy available cards based on player's coins 2. end treasure phase
            ACTION_PHASE: player actions in your card
        """

    def __str____(self):
        """String for debug purposes. No particular semantics are required."""
        pass

    def apply_action(self, action):
        pass

class DominionObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        pass


    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        pass


    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pass

pyspiel.register_game(_GAME_TYPE, DominionGame)
