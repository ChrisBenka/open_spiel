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

import random
from operator import itemgetter
import numpy as np
import pyspiel

_MIN_PLAYERS = 2
_MAX_PLAYERS = 4
_HAND_SIZE = 5
_DRAW_PILE_SIZE = 10
_NUM_KINGDOM_SUPPLY_PILES = 10


_DEFAULT_PARAMS = {
    'num_players': _MIN_PLAYERS,
    'automate_action_phase': True
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_dominion",
    long_name="Python Dominion",
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
    def __init__(self, name: str, cost: int):
        self.name = name
        self.cost = cost
    def __eq__(self,other):
        return self.name == other.name
  
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
    def __init__(self, name: str, coins: int, cost: int, effect_list: list, add_cards: int, global_trigger: callable):
        super(AttackCard, self).__init__(name=name, coins=coins, cost=cost, effect_list=effect_list,
                                         add_cards=add_cards)
        self.global_trigger = global_trigger

class ReactionCard(ActionCard):
    def __init__(self, name: str, cost: int, add_cards: int, global_trigger: callable):
        super(ReactionCard, self).__init__(name, cost, add_cards=add_cards)
        self.global_trigger = global_trigger
    def __str__(self):
        return str(self.card)

class SupplyPile:
    def __init__(self, card, qty):
        self.card = card
        self.qty = qty

""" KINGDOM CARDS """
VILLAGE = ActionCard(name='Village', cost=3, add_actions=2, add_cards=1)
LABORATORY = ActionCard(name='Laboratory', cost=5, add_cards=2, add_actions=1)
MARKET = ActionCard(name='Market', cost=5, add_actions=1, add_buys=1, coins=1, add_cards=1)
FESTIVAL = ActionCard(name='Festival', cost=5, add_actions=2, add_buys=1, coins=2)
SMITHY = ActionCard(name="Smithy", cost=4, add_cards=3)
MILITIA = AttackCard(name='Militia', cost=4, coins=2, effect_list=[], add_cards=0, global_trigger=None)
GARDENS = VictoryCard(name='Gardens', cost=4, victory_points=0,vp_fn=lambda all_cards: math.floor(len(all_cards) / 10))
CHAPEL = ActionCard(name='Chapel', cost=2, effect_list=[])
WITCH = AttackCard(name='Witch', cost=5, add_cards=2, effect_list=[], coins=0, global_trigger=None)
WORKSHOP = ActionCard(name='Workshop', cost=3, effect_list=[])
BANDIT = AttackCard(name='Bandit', cost=5, effect_list=[], coins=0, add_cards=0, global_trigger=None)
REMODEL = ActionCard(name='Remodel', cost=4, effect_list=[])
THRONE_ROOM = ActionCard(name='Throne Room', cost=4, effect_fn=None)
MONEYLENDER = ActionCard(name='Moneylender', cost=4, effect_fn=None)
POACHER = ActionCard(name='Poacher', cost=4, add_cards=4, add_actions=1, coins=1, effect_fn=None)
MERCHANT = ActionCard(name='Merchant', cost=3, add_cards=1, add_actions=1, effect_fn=None)
CELLAR = ActionCard(name='Cellar', cost=2, effect_fn=None)
MINE = ActionCard(name='Mine', cost=5, effect_fn=None)
VASSAL = ActionCard(name='Vassal', cost=3, coins=2, effect_fn=None)
COUNCIL_ROOM = ActionCard(name='Council Room', cost=5, add_cards=4, add_buys=1, effect_fn=None)
ARTISAN = ActionCard(name='Artisan', cost=6, effect_fn=None)
BUREAUCRAT = ActionCard(name='Bureaucrat', cost=4, effect_fn=None)
SENTRY = ActionCard(name='Sentry', cost=5, add_cards=1, add_actions=1, effect_fn=None)
HARBINGER = ActionCard(name='Harbinger', cost=3, add_cards=1, add_actions=1, effect_list=[])
LIBRARY = ActionCard(name='Library', cost=5, effect_fn=None)
MOAT = ReactionCard(name='Moat', cost=2, add_cards=2, global_trigger=None)

""" TREASURE CARDS """
COPPER = TreasureCard(name='Copper', cost=0, coins=1)
SILVER = TreasureCard(name='Silver', cost=3, coins=2)
GOLD = TreasureCard(name='Gold', cost=6, coins=3)

""" VICTORY CARDS """ 
CURSE = VictoryCard(name='Curse', cost=0, victory_points=-1)
DUCHY = VictoryCard(name='Duchy', cost=4, victory_points=5)
ESTATE = VictoryCard(name='Estate', cost=2, victory_points=1)
PROVINCE = VictoryCard(name='Province', cost=8, victory_points=8)

_INIT_KINGDOM_SUPPLY = {
    "Village": SupplyPile(VILLAGE, 10),
    "Laboratory": SupplyPile(LABORATORY, 10),
    "Market": SupplyPile(MARKET, 10),
    "Festival": SupplyPile(FESTIVAL, 10),
    "Smithy": SupplyPile(SMITHY, 10),
    "Militia": SupplyPile(MILITIA, 10),
    "Gardens": SupplyPile(GARDENS, 8),
    "Chapel": SupplyPile(CHAPEL, 10),
    "Witch": SupplyPile(WITCH, 10),
    "Workshop": SupplyPile(WORKSHOP, 10),
    "Bandit": SupplyPile(BANDIT, 10),
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
    "Moat": SupplyPile(MOAT, 10),
}
_INIT_TREASURE_CARDS_SUPPLY = {
    'Copper': SupplyPile(COPPER, 46), 
    'Silver': SupplyPile(SILVER, 40),
    'Gold': SupplyPile(GOLD, 30)
}

_INIT_VICTORY_CARDS_SUPPLY = {
    'Curse': SupplyPile(CURSE, 10), 
    'Estate': SupplyPile(ESTATE, 8),                          
    'Duchy': SupplyPile(DUCHY, 8), 
    'Province': SupplyPile(PROVINCE, 8)
}

NUM_KINGDOM_PILES = _NUM_KINGDOM_SUPPLY_PILES
NUM_TREASURE_PILES = len(_INIT_TREASURE_CARDS_SUPPLY.keys())
NUM_VICTORY_PILES = len(_INIT_VICTORY_CARDS_SUPPLY.keys())
NUM_UNIQUE_CARDS = NUM_KINGDOM_PILES + NUM_TREASURE_PILES + NUM_VICTORY_PILES

END_PHASE_ACTION = NUM_UNIQUE_CARDS

  
_PRESET_KINGDOM_CARDS = [MOAT,VILLAGE,BUREAUCRAT,SMITHY,MILITIA,WITCH,LIBRARY,MARKET,MINE,COUNCIL_ROOM]
_PRESET_KINGDOM_CARDS_NAMES = list(map(lambda card: card.name, _PRESET_KINGDOM_CARDS))

_TREASURE_CARDS = [COPPER,GOLD,SILVER]
_TREASURE_CARDS_NAMES = list(map(lambda card: card.name, _TREASURE_CARDS))

_VICTORY_CARDS = [CURSE,ESTATE,DUCHY,PROVINCE]
_VICTORY_CARDS_NAMES = list(map(lambda card: card.name, _VICTORY_CARDS))


class TurnPhase(enumerate):
    ACTION_PHASE = 1
    TREASURE_PHASE = 2
    BUY_PHASE = 3
    END_TURN = 4

class Move(object):
    """
    An Move is an Action that a player can take.

    Must implement do(game_state), which is called when the player selects that Move.
    """
    def do(self, state):
        raise Exception("Move does not implement do.")


class Player(object):
    def __init__(self, id):
        self.id = id
        self.victory_points = 0
        self.draw_pile = [COPPER for _ in range(7)] + [ESTATE for _ in range(3)]
        self.discard_pile = []
        self.trash_pile = []
        self.hand = []
        self.phase = TurnPhase.END_TURN
        self.actions = 0
        self.buys = 0
        self.coins = 0

    def _draw_hand(self):
        """ draw a player's hand (5 cards) from their draw pile """
        """
        if there are not enough cards in draw pile, draw as many as he can and 
        shuffle discard pile to form new draw pile
        """
        cards_drawn_idxs = set(random.sample(range(len(self.draw_pile)), _HAND_SIZE - len(self.hand)))
        self.hand = list(itemgetter(*cards_drawn_idxs)(self.draw_pile))
        self.draw_pile = [card for idx, card in enumerate(self.draw_pile) if not idx in cards_drawn_idxs]
    
    def play_card_from_hand(self,card: Card):
        self.coins += card.coins or 0
        self.hand.remove(card)
 

    def init_turn(self):
        self._draw_hand()
        num_action_cards = len(list(filter(lambda card: type(Card) == 'ActionCard',self.hand)))
        self.actions = 1 if num_action_cards > 0 else 0 
        self.buys = 1
        self.coins = 0
        self.phase = TurnPhase.ACTION_PHASE if num_action_cards > 0 else TurnPhase.TREASURE_PHASE
    
    def end_phase(self):
        if self.phase is TurnPhase.ACTION_PHASE:
            self.phase = TurnPhase.TREASURE_PHASE
        elif self.phase is TurnPhase.TREASURE_PHASE:
            self.phase = TurnPhase.BUY_PHASE
        elif self.phase is TurnPhase.BUY_PHASE:
            self.phase = TurnPhase.END_TURN
        return self.phase

    def end_turn(self):
        # cleanup-phase
        pass

    def get_state(self):
        return self.draw_pile, self.victory_points, self.hand, self.discard_pile, self.trash_pile


class DominionGame(pyspiel.Game):
    """ A python version of Dominion."""

    def __init__(self, params=None):
        self._GAME_INFO = pyspiel.GameInfo(
            num_distinct_actions=NUM_UNIQUE_CARDS,
            max_chance_outcomes=0,
            num_players=params["num_players"],
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0,
            max_game_length=3
        )
        # self.kingdom_cards = _PRESET_KINGDOM_CARDS
        # self.treasure_cards = _TREASURE_CARDS
        # self.victory_cards = _VICTORY_CARDS
        super().__init__(_GAME_TYPE, self._GAME_INFO, params or dict())

    def get_game_info(self):
        return self._GAME_INFO

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return DominionGameState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return DominionObserver(iig_obs_type, {"num_players": self.get_game_info().num_players})


class DominionGameState(pyspiel.State):
    """ a python version of the Dominon state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)

        self._players = [Player(id=i) for i in range(game.num_players())]

        '''
        state
        this info is also stored in each respective player
        making data redundant to avoid need to loop through each player to get game state
        '''
        self.draw_piles = [player.draw_pile for player in self._players]
        self.victory_points = [player.victory_points for player in self._players]
        self.hands = [player.hand for player in self._players]
        self.discard_piles = [player.discard_pile for player in self._players]
        self.trash_piles = [player.trash_pile for player in self._players]
        self.kingdom_piles = {key: _INIT_KINGDOM_SUPPLY[key] for key in _INIT_KINGDOM_SUPPLY if
                              key in _PRESET_KINGDOM_CARDS_NAMES}
        self.treasure_piles = _INIT_TREASURE_CARDS_SUPPLY
        self.victory_piles = _INIT_VICTORY_CARDS_SUPPLY
        self._all_supply_piles = list(self.treasure_piles.values()) + list(self.victory_piles.values()) + list(self.kingdom_piles.values())
        self._cur_player = 0
        self._is_terminal = False

        self._start_game()

    def _start_game(self):
        self._players[self._cur_player].init_turn()
    
    def is_terminal(self):
        no_provinces_left = self.victory_piles[PROVINCE.name].qty is 0
        three_piles_empty = len(list(filter(lambda supply: supply.qty is 0,self._all_supply_piles))) is 3
        self._is_terminal = no_provinces_left or three_piles_empty
        return self._is_terminal

    def _update_game_state_from_player(self):
        player_id = self._cur_player
        self.draw_piles[player_id], self.victory_points[player_id], self.hands[player_id], self.discard_piles[player_id]
        self.trash_piles[player_id] = self._players[player_id].getState()

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player_id: int) -> list:
        """ treasure_cards, victory_cards, kindom_cards"""
        player = self.get_player(player_id)
        if player.phase is TurnPhase.TREASURE_PHASE:
            """ player can play their treasure cards in exchange for coins and end current phase"""
            unique_cards_in_hand = set(map(lambda card: card.name,player.hand))
            return [idx for idx,treasure_card in enumerate(_TREASURE_CARDS_NAMES) if treasure_card in unique_cards_in_hand] + [END_PHASE_ACTION]
        elif player.phase is TurnPhase.BUY_PHASE:
            """ player can buy any card whose buy value is less than or equal to player's coins and card supply > 0; end current phase""" 
            is_valid_card_to_buy = lambda supply_tuple : supply_tuple[1].qty > 0 and supply_tuple[1].card.cost <= player.coins
            get_idx = lambda supply_tuple: supply_tuple[0]
            all_valid_cards = list(map(get_idx,filter(is_valid_card_to_buy,enumerate(self._all_supply_piles))))
            return all_valid_cards + [END_PHASE_ACTION]
        elif player.phase is TurnPhase.ACTION_PHASE:
            return []

    def _action_to_string(self,player,action) -> str:
        """Action -> string."""
        if action is END_PHASE_ACTION:
            return "End phase"
        phase = "Buy and Gain" if self.get_player(player).phase is TurnPhase.BUY_PHASE else "Play"
        return "{} {}".format(phase,self._all_supply_piles[action].card.name)

    def get_player(self, id):
        return self._players[id]

    def get_players(self) -> list:
        return self._players

    def __str____(self):
        """String for debug purposes. No particular semantics are required."""
        pass

    def play_treasure_card(self,card: TreasureCard):
        player = self.get_player(self.current_player())
        player.play_card_from_hand(card)
        self.treasure_piles[card.name].qty -= 1
        all_treasure_cards_played = len(list(filter(lambda card: card.name in _TREASURE_CARDS_NAMES,player.hand))) == 0
        if all_treasure_cards_played:
            player.end_phase()

    def play_end_phase(self):
        uptd_phase = self.get_player(self.current_player()).end_phase()
        if uptd_phase is TurnPhase.END_TURN:
            self.player.end_turn()
            self.move_to_next_player()
  
    def apply_action(self, action):
        if self.is_terminal():
            raise Exception("Game is finished")
        player = self._cur_player
        legal_actions = self._legal_actions(player)
        if action not in legal_actions:
            action_str = lambda action: f"{action}:{self._action_to_string(player,action)}"
            legal_actions_str = ", ".join(list(map(action_str,legal_actions)))
            raise Exception(f"Action {action_str(action)} not in list of legal actions - {legal_actions_str}")
        else:
            if action is not END_PHASE_ACTION:
                self.play_treasure_card(self._all_supply_piles[action].card)
            else:
                self.play_end_phase()
            
    def move_to_next_player(self):
        if not self.is_terminal():
            self._cur_player = (self._curr_player + 1) % len(self._players)
            self._players[self._cur_player].init_turn()
        else:
            raise Exception("Game is finished")


class DominionObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""        
        # different components of observation
        pieces = [
            ("kingdom_piles", _NUM_KINGDOM_SUPPLY_PILES, (_NUM_KINGDOM_SUPPLY_PILES,)),
            ("treasure_piles", NUM_TREASURE_PILES, (NUM_TREASURE_PILES,)),
            ("victory_piles", NUM_VICTORY_PILES, (NUM_VICTORY_PILES,)),
            ("victory_points", params["num_players"], (params["num_players"],)),
            ('TurnPhase',1,(1,)),
            ('actions', 1, (1,)),
            ('buys', 1, (1,)),
            ('coins', 1, (1,)),
            ('draw',NUM_UNIQUE_CARDS,(NUM_UNIQUE_CARDS,)),
            ('hand',NUM_UNIQUE_CARDS,(NUM_UNIQUE_CARDS,)),
            ('discard',NUM_UNIQUE_CARDS,(NUM_UNIQUE_CARDS,)),
            ('trash',NUM_UNIQUE_CARDS,(NUM_UNIQUE_CARDS,))            
        ]

        # build the single flat tensor
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.int32)

        # build the named & reshaped view of the components of the flat tensor
        self.dict = {}
        idx = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[idx:idx + size].reshape(shape)
            idx += size
    
    def _count_cards(self,cards):
        return np.unique(list(map(lambda card: card.name,cards)),return_counts=True)

    def _num_all_cards(self,pile:list):
        """treasure_cards,victory_cards,kingdom_cards"""
        num_cards = dict.fromkeys(_TREASURE_CARDS_NAMES+_VICTORY_CARDS_NAMES+_PRESET_KINGDOM_CARDS_NAMES,0)
        cards,nums = self._count_cards(pile)
        for card,num in zip(cards,nums):
            num_cards[card] = num
        return list(num_cards.values())

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        idx = 0
        kingdom_piles = [pile.qty for pile in state.kingdom_piles.values()]
        treasure_piles = [pile.qty for pile in state.treasure_piles.values()]
        victory_piles = [pile.qty for pile in state.victory_piles.values()]
        victory_points = [player.victory_points for player in state.get_players()]

        
        values = [
            ('kingdom_piles', kingdom_piles),
            ('treasure_piles', treasure_piles),
            ('victory_piles', victory_piles),
            ('victory_points', state.victory_points),
            ('TurnPhase',[state.get_player(player).phase]),
            ('actions', [state.get_player(player).actions]),
            ('buys', [state.get_player(player).buys]),
            ('coins', [state.get_player(player).coins]),
            ('draw',self._num_all_cards(state.get_player(player).draw_pile)),
            ('hand',self._num_all_cards(state.get_player(player).hand)),
            ('discard',self._num_all_cards(state.get_player(player).discard_pile)),
            ('trash',self._num_all_cards(state.get_player(player).trash_pile))
        ]

        for name, value in values:
            self.dict[name] = value
            self.tensor[idx: idx + len(value)] = value
            idx += len(value)

    def _string_count_cards(self,cards):
        unique_cards,num_unique = self._count_cards(cards)
        return ", ".join([f"{card}: {qty}" for card,qty in  zip(unique_cards,num_unique)])

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        pieces.append(f"p{player}: ")
        kingdom_supply_piles = ", ".join([f"{item[0]}: {item[1].qty}" for item in state.kingdom_piles.items()])
        treasure_piles = ", ".join([f"{item[0]}: {item[1].qty}" for item in state.treasure_piles.items()])
        victory_piles = ", ".join([f"{item[0]}: {item[1].qty}" for item in state.victory_piles.items()])
        victory_points = ", ".join([f"p{player}: {vp}" for player,vp in enumerate(state.victory_points)])
        action = ", ".join([f"p{player}: {vp}" for player,vp in enumerate(state.victory_points)])
        buys = ", ".join([f"p{player}: {vp}" for player,vp in enumerate(state.victory_points)])
        coins = ", ".join([f"p{player}: {vp}" for player,vp in enumerate(state.victory_points)])

        

        pieces.append(f"kingdom supply piles: {kingdom_supply_piles}")
        pieces.append(f"treasure supply piles: {treasure_piles}")
        pieces.append(f"victory supply piles: {victory_piles}")
        pieces.append(f"victory points: {victory_points}")
        pieces.append(f"Turn Phase: {state.get_player(player).phase}")
        pieces.append(f"actions: {state.get_player(player).actions}")
        pieces.append(f"buys: {state.get_player(player).buys}")
        pieces.append(f"coin: {state.get_player(player).coins}")
        draw_pile = self._string_count_cards(state.get_player(player).draw_pile)
        pieces.append(f"draw pile: {draw_pile if len(draw_pile) > 0 else 'empty'}")
        hand = self._string_count_cards(state.get_player(player).hand)
        pieces.append(f"hand: {hand if len(hand) > 0 else 'empty'}")
        discard_pile = self._string_count_cards(state.get_player(player).discard_pile)
        pieces.append(f"discard pile: {discard_pile if len(discard_pile) > 0 else 'empty'}")
        trash_pile = self._string_count_cards(state.get_player(player).trash_pile)
        pieces.append(f"trash pile: {trash_pile if len(trash_pile) > 0 else 'empty'}")

        return "\n".join(str(p) for p in pieces)

# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, DominionGame)
