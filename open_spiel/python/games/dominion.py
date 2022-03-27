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

import math
import random

import numpy as np
from collections import Counter 
import pyspiel

_MIN_PLAYERS = 2
_MAX_PLAYERS = 4
_HAND_SIZE = 5
_DRAW_PILE_SIZE = 10
_NUM_KINGDOM_SUPPLY_PILES = 10

_DEFAULT_PARAMS = {
    'num_players': _MIN_PLAYERS,
    'automate_action_phase': True,
    'verbose': True,
    'kingdom_cards': "Moat, Village, Bureaucrat, Smithy, Militia, Witch, Library, Market, Mine, Council Room"
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

PLAY_ID = 1
BUY_ID = 34
DISCARD_ID = 67
TRASH_ID = 100
GAIN_ID = 133

class GameFinishedException(Exception):
    pass

class Card(object):
    def __init__(self, id: int, name: str, cost: int):
        self.name = name
        self.cost = cost
        self.id = id
        
        self.play = None
        self.buy = None
        self.discard = None
        self.trash = None
        self.gain = None

        self.__set_actions()
    
    def __hash__(self):
        return hash(self.id)

    def __set_actions(self):
        global PLAY_ID
        global BUY_ID
        global TRASH_ID
        global DISCARD_ID
        global GAIN_ID

        self.play = PLAY_ID
        self.buy = BUY_ID
        self.discard = DISCARD_ID
        self.trash = TRASH_ID
        self.gain = GAIN_ID

        PLAY_ID += 1
        BUY_ID += 1
        DISCARD_ID += 1
        TRASH_ID += 1
        GAIN_ID  += 1

        self.action_strs = {self.play: "Play", self.buy: "Buy", self.discard: "Discard", self.trash: "Trash", self.gain: "Gain"}

    def __eq__(self, other):
        return self.name == other.name
    
    def __str__(self):
        return self.name

    def _action_to_string(self,action):
        return f"{self.action_strs[action]} {self.name}"


class VictoryCard(Card):
    def __init__(self, id: int, name: str, cost: int,
                 victory_points: int, vp_fn: callable = None):
        super(VictoryCard, self).__init__(id, name, cost)
        self.victory_points = victory_points
        self.vp_fn = vp_fn


class TreasureCard(Card):
    def __init__(self, id: int, name: str, cost: int, coins: int):
        super(TreasureCard, self).__init__(id, name, cost)
        self.coins = coins


class ActionCard(Card):
    def __init__(self, id: int, name: str, cost: int,
                 add_actions: int = 0, add_buys: int = 0, add_cards: int = 0,
                coins: int = 0, effect_list: list = []):
        super(ActionCard, self).__init__(id, name, cost)
        self.add_actions = add_actions
        self.add_buys = add_buys
        self.add_cards = add_cards
        self.effect_list = effect_list
        self.coins = coins


class AttackCard(ActionCard):
    def __init__(self, id: int, name: str, coins: int, cost: int,
                 effect_list: list, add_cards: int):
        super(AttackCard, self).__init__(id, name=name, coins=coins, cost=cost, effect_list=effect_list,
                                         add_cards=add_cards)


class ReactionCard(ActionCard):
    def __init__(self, id: int, name: str, cost: int, add_cards: int):
        super(ReactionCard, self).__init__(id=id, name=name, cost=cost, add_cards=add_cards)


class SupplyPile:
    def __init__(self, card, qty):
        self.card = card
        self.qty = qty


class Effect(object):
    def run(self, state, player):
        raise Exception("Effect does not implement run!")


class TrashCardsEffect(Effect):
    """
    Player trashes n cards Example: Chapel.
    """

    def __init__(self, num_cards,name,filter_func=None, optional=True):
        self.id = 1
        self.num_cards = num_cards
        self.filter_func = filter_func
        self.optional = optional
        self.name = name
        self.num_cards_trashed = 0

    def __eq__(self, other):
        return self.num_cards == other.num_cards and self.filter_func == other.filter_func and self.optional == other.optional

    def _legal_actions(self, state, player):
        is_valid_card_to_trash = lambda card: card in player.hand
        trashable_cards = list(map(lambda card: card.trash, filter(is_valid_card_to_trash, _ALL_CARDS)))
        return trashable_cards + [END_PHASE_ACTION]
    
    def __str__(self):
        return f"Trash up to {self.num_cards - self.num_cards_trashed} cards"
    
    def _action_to_string(self,action):
        card = _get_card(action)
        return card._action_to_string(action) if action is not END_PHASE_ACTION  else "End trash effect"

    def _apply_action(self, state, action):
        player = state.get_current_player()
        if action is END_PHASE_ACTION:
            state.effect_runner.effects[player.id] = None
        else:
            id = (action-1) % len(_ALL_CARDS)
            card = _ALL_CARDS[id]
            player.hand.remove(card)
            player.trash_pile.append(card)
            self.num_cards_trashed += 1
            if self.num_cards_trashed == self.num_cards:
                state.effect_runner.effects[player.id] = None


    def run(self, state, player):
        state.effect_runner.initiator = player.id
        state.effect_runner.add_effect(player.id, self)

class DiscardDownToEffect(Effect):
    """
    Discard down to some number of cards in player's hand. Example: Militia.
    """
    def __init__(self, num_cards_downto):
        self.id = 2
        self.num_cards_downto = num_cards_downto
        
    def __str__(self):
        return f"Discard {self.num_cards_downto} cards"

    def __eq__(self, other):
        return self.num_cards_downto == other.num_cards_downto

    def _legal_actions(self, state, player):
        is_valid_card_to_discard = lambda card: card in player.hand
        discardable_cards = list(map(lambda card: card.discard, filter(is_valid_card_to_discard, _ALL_CARDS)))
        return discardable_cards

    def _action_to_string(self,action):
        card = _ALL_CARDS[(action-1) % len(_ALL_CARDS)]
        return card._action_to_string(action)

    def _apply_action(self, state, action):
        id = (action-1) % len(_ALL_CARDS)
        card = _ALL_CARDS[id]
        player = state.get_current_player()
        player.hand.remove(card)
        player.discard_pile.append(card)
        if len(player.hand) is self.num_cards_downto:
            state.effect_runner.effects[player.id] = None
    
    def run(self,state,player):
        state.effect_runner.add_effect(player.id,self)

class OpponentsDiscardDownToEffect(Effect):
    def __init__(self, num_cards_downto):
        self.id = 3
        self.num_cards_downto = num_cards_downto
    def _legal_actions(self, state, player):
        """ the only legal action is to play a moat or not play a moat """
        return [MOAT.play,END_PHASE_ACTION]
    
    def _apply_action(self,state,action):
        if action is END_PHASE_ACTION:
            effect = DiscardDownToEffect(self.num_cards_downto)
            effect.run(state,state.current_player())
        else:
            state.effect_runner.effects[state.current_player()] = None

    def _action_to_string(self,action):
        if action is END_PHASE_ACTION:
            return f"Do not play Moat and discard {self.num_cards_downto} cards"
        else:
            card = _get_card(action)
            return f"{card._action_to_string(action)} and do not discard {self.num_cards_downto} cards"

    def run(self, state, player):
        state.effect_runner.initiator = player.id
        for opp in state.other_players(player):
            if MOAT in state.get_player(opp).hand:
                state.effect_runner.add_effect(opp,self)
            else:
                effect = DiscardDownToEffect(self.num_cards_downto)
                effect.run(state,state.get_player(opp))


class GainCardToDiscardPileEffect(Effect):
    def __init__(self, card):
        self.card = card
        self.id = 4

    def __eq__(self, other):
        return self.card == other.card
    
    def __str__(self):
        return f"Gain {self.card.name} to discard pile"

    def run(self, state, player):
        state.supply_piles[self.card.name].qty -= 1
        player.discard_pile.append(self.card)


class OpponentsGainCardEffect(Effect):
    def __init__(self, card):
        self.card = card
        self.id = 5
    def __eq__(self, other):
        return self.card == other.card
    def _legal_actions(self, state, player):
        """ the only legal action is to play a moat or not play a moat """
        return [MOAT.play,END_PHASE_ACTION]
    
    def _action_to_string(self,action):
        if action is END_PHASE_ACTION:
            return f"Do not play Moat and gain {self.card}"
        else:
            card = _get_card(action)
            return f"{card._action_to_string(action)} and do not gain {self.card}"
    
    def _apply_action(self, state, action):
        if action is END_PHASE_ACTION:
            state.effect_runner.effects[state.current_player()] = None
            effect = GainCardToDiscardPileEffect(self.card)
            effect.run(state,state.get_current_player())
        else:
            state.effect_runner.effects[state.current_player()] = None

    def run(self, state, player):
        for opp in state.other_players(player):
            if MOAT in state.get_player(opp).hand:
                state.effect_runner.initiator = state.current_player()
                state.effect_runner.add_effect(opp,self)
            else:
                effect = GainCardToDiscardPileEffect(self.card)
                effect.run(state,state.get_player(opp))

class OpponentsGainCardToHandEffect(Effect):
    def __init__(self, num_cards=1):
        self.num_cards = num_cards
        self.id = 6
    def __eq__(self, other):
        return self.num_cards == other.num_cards

    def run(self, state, player):
        for opp in state.other_players(player):
            state.get_player(opp)._draw_hand(1)


class ChoosePileToGainEffect(Effect):
    """Choose a pile to gain a card from. Card's cost must <= than n coins. E.g. Workshop"""

    def __init__(self, n_coins):
        self.n_coins = n_coins
        self.id = 7

    def __eq__(self, other):
        return self.n_coins == other.n_coins

    def _legal_actions(self, state, player):
        is_valid_to_gain = lambda supply_pile: supply_pile.qty > 0 and supply_pile.card.cost <= self.n_coins
        gainable_cards = list(
            map(lambda pile: pile.card.gain, filter(is_valid_to_gain, state.supply_piles.values())))
        gainable_cards.sort()
        return gainable_cards

    def __str__(self):
        return f"Choose a pile whose cost is <= {self.n_coins}"

    def _apply_action(self, state, action):
        player = state.get_current_player()
        card = _get_card(action)
        state.supply_piles[card.name].qty -= 1
        player.discard_pile.append(card)
        state.effect_runner.effects[player.id] = None
    
    def _action_to_string(self,action):
        card = _get_card(action)
        return f"{card._action_to_string(action)} to discard pile"

    def run(self, state, player):
        state.effect_runner.initiator = player.id
        state.effect_runner.add_effect(player.id, self)


class TrashAndGainCostEffect(Effect):
    """trash a card from your hand. gain a card costing up to 2 more than it. E.g. remodel"""

    def __init__(self, add_cost: int, gain_exact_cost: bool):
        self.add_cost = add_cost
        self.gain_exact_cost = gain_exact_cost
        self.has_trashed = False
        self.trashed_card = None
        self.id = 8


    def __eq__(self, other):
        return self.add_cost is other.add_cost and self.gain_exact_cost is other.gain_exact_cost

    def _legal_actions(self, state, player):
        if self.trashed_card is None:
            is_valid_card_to_trash = lambda card: card in player.hand
            trashable_cards = list(map(lambda card: card.trash, filter(is_valid_card_to_trash, _ALL_CARDS)))
            return trashable_cards
        else:
            is_valid_card_to_gain_exact = lambda \
                    pile: pile.qty > 0 and pile.card.cost == self.trashed_card.cost + self.add_cost
            is_valid_card_to_gain = lambda \
                    pile: pile.qty > 0 and pile.card.cost <= self.trashed_card.cost + self.add_cost
            gainable_cards = filter(is_valid_card_to_gain_exact,
                                    state.supply_piles.values()) if self.gain_exact_cost else filter(
                is_valid_card_to_gain, state.supply_piles.values())
            gainable_cards = list(map(lambda pile: pile.card.gain, gainable_cards))
            gainable_cards.sort()
            return gainable_cards
    
    def __str__(self):
        return f"Trash a card from hand, gain a card costing up to more than {self.add_cost} the card trashed"

    def _apply_action(self, state, action):
        player = state.get_current_player()
        if self.trashed_card is None:
            card = _get_card(action)
            player.hand.remove(card)
            player.trash_pile.append(card)
            self.trashed_card = card
        else:
            card = _get_card(action)
            state.supply_piles[card.name].qty -= 1
            player.discard_pile.append(card)
            state.effect_runner.effects[player.id] = None

    def _action_to_string(self,action):
        card = _get_card(action)
        return card._action_to_string(action)

    def run(self, state, player):
        state.effect_runner.initiator = player.id
        state.effect_runner.add_effect(player.id, self)


class TrashTreasureAndGainCoinEffect(Effect):
    """trash a treasure card from your hand. gain n coins E.g. moneylender"""

    def __init__(self, treasure_card: TreasureCard, n_coins: int, optional_trash=True):
        self.treasure_card = treasure_card
        self.n_coins = n_coins
        self.optional_trash = optional_trash
        self.id = 9
    def __eq__(self, other):
        return self.treasure_card == other.treasure_card and \
               self.n_coins == other.n_coins and \
               self.optional_trash == other.optional_trash

    def _legal_actions(self, state, player):
        return [self.treasure_card.trash, END_PHASE_ACTION] if self.optional_trash else [self.treasure_card.trash]
    
    def __str__(self):
        return f"Trash {self.treasure_card.name} from hand. Gain {self.n_coins} coins"

    def _apply_action(self, state, action):
        # if user decides not to trash the treasure_card from their hand do nothing
        player = state.get_current_player()
        if action is END_PHASE_ACTION:
            state.effect_runner.effects[player.id] = None
            return
        card = _get_card(action)
        player.trash_pile.append(card)
        player.hand.remove(card)
        player.coins += self.n_coins
        state.effect_runner.effects[player.id] = None
    
    def _action_to_string(self,action):
        if action is END_PHASE_ACTION:
            return f"End Trash {self.treasure_card.name}"
        card = _get_card(action)
        return card._action_to_string(action)

    def run(self, state, player):
        # if player does not have the treasure card do nothing
        if self.treasure_card in player.hand:
            state.effect_runner.initiator = player.id
            state.effect_runner.add_effect(player.id, self)


class PoacherEffect(Effect):
    """Discard a card per empty Supply pie.""" 
    def __init__(self):
        self.id = 10

    def __eq__(self, other):
        return True

    def _legal_actions(self, state, player):
        return list(set(list(map(lambda card: card.discard, player.hand))))
    

    def _apply_action(self, state, action):
        player = state.get_current_player()
        card = _get_card(action)
        player.hand.remove(card)
        player.discard_pile.append(card)
        if len(player.hand) == self.num_empty_supply_piles:
            state.effect_runner.effects[player.id] = None

    def run(self, state, player):
        self.num_empty_supply_piles = len(list(filter(lambda pile: pile.qty == 0, state.supply_piles.values())))
        if self.num_empty_supply_piles == 0:
            return
        else:
            state.effect_runner.initiator = player.id
            state.effect_runner.add_effect(player.id, self)


class CellarEffect(Effect):
    """+1 Action. Discard any number of cards, then draw that many."""

    def __init__(self):
        self.id = 11
        self.num_cards_discarded = 0
        self.cards_to_discard = []

    def __eq__(self, other):
        return True

    def _legal_actions(self, state, player):
        return list(set(list(map(lambda card: card.discard, player.hand)))) + [END_PHASE_ACTION]

    def _apply_action(self, state, action):
        if action is END_PHASE_ACTION:
            player = state.get_current_player()
            player.discard_pile += self.cards_to_discard
            player._draw_hand(self.num_cards_discarded)
            state.effect_runner.effects[player.id] = None
        else:
            card = _get_card(action)
            player = state.get_current_player()
            player.hand.remove(card)
            self.num_cards_discarded += 1
            self.cards_to_discard.append(card)
    
    def _action_to_string(self,action):
        if action is END_PHASE_ACTION:
            return "End discard"
        else:
            card = _get_card(action)
            return card._action_to_string(action)

    def run(self, state, player):
        state.effect_runner.initiator = player.id
        state.effect_runner.add_effect(player.id, self)


class TrashTreasureAndGainTreasure(Effect):
    """Trash and Gain a Treasure from your hand. Gain a treasure to your hand costing up to n_coins more than it"""

    def __init__(self, n_coins):
        self.n_coins = n_coins
        self.trashed_card = None
        self.id = 12

    def __eq__(self, other):
        return self.n_coins == other.n_coins

    def _legal_actions(self, state, player):
        if self.trashed_card is None:
            actions = list(set(list(
                map(lambda card: card.trash, filter(lambda card: isinstance(card, TreasureCard), player.hand))))) + [
                       END_PHASE_ACTION]
            return sorted(actions)
        else:
            gainable_treasure_cards = list(map(lambda card_nm: state.supply_piles[card_nm].card.gain, filter(
                lambda treasure_card_nm: state.supply_piles[treasure_card_nm].qty > 0 and state.supply_piles[treasure_card_nm].card.cost <= self.trashed_card.cost + self.n_coins,
                _TREASURE_CARDS_NAMES)))
            return sorted(gainable_treasure_cards)

    def _apply_action(self, state, action):
        if self.trashed_card is None:
            if action is END_PHASE_ACTION:
                player = state.get_current_player()
                state.effect_runner.effects[player.id] = None
            else:
                card = _get_card(action)
                player = state.get_current_player()
                player.hand.remove(card)
                player.trash_pile.append(card)
                self.trashed_card = card
        else:
            card = _get_card(action)
            player = state.get_current_player()
            player.hand.append(card)
            state.effect_runner.effects[player.id] = None
    
    def _action_to_string(self,action):
        if action is END_PHASE_ACTION:
            return "End trash treasure"
        else:
            card = _get_card(action)
            return card._action_to_string(action)
    def run(self, state, player):
        has_treasure_card = len(list(filter(lambda card: isinstance(card, TreasureCard), player.hand)))
        if has_treasure_card:
            state.effect_runner.initiator = player.id
            state.effect_runner.add_effect(player.id, self)


class VassalEffect(Effect):
    """If the top of your draw pile is an action, vassal can play it; otherwise, the top card is discarded"""

    def __init__(self):
        self.top_of_deck = None
        self.id = 13
    def __eq__(self, other):
        return True

    def _legal_actions(self, state, player):
        return [self.top_of_deck.play, END_PHASE_ACTION]

    def _apply_action(self, state, action):
        player = state.get_current_player()
        if action is END_PHASE_ACTION:
            player.discard_pile.append(self.top_of_deck)
            state.effect_runner.effects[player.id] = None
        else:
            card = _get_card(action)
            # player will have played 1 action card by playing the vassal, and will lose 1 action for playing the ActionCard. Need to add 2 to offset this
            player.actions += 2
            player.hand.append(card)
            state.play_action_card(card)
            state.effect_runner.effects[player.id] = None

    def _action_to_string(self,action):
        pass
        
    def run(self, state, player):
        if len(player.draw_pile) == 0:
            player._add_discard_pile_to_draw_pile()
        self.top_of_deck = player.draw_pile.pop(0)
        if isinstance(self.top_of_deck, ActionCard):
            state.effect_runner.initiator = player.id
            state.effect_runner.add_effect(player.id, self)
        else:
            player.discard_pile.append(self.top_of_deck)


class ArtisanEffect(Effect):
    def __init__(self):
        self.n_coins = 5
        self.card_to_gain = None
        self.name = "Artisan"
        self.id = 14

    def _legal_actions(self, state, player):
        if self.card_to_gain is None:
            gainable_cards = list(map(lambda pile: pile.card.gain,
                                      filter(lambda pile: pile.qty > 0 and pile.card.cost <= self.n_coins,
                                             state.supply_piles.values())))
            gainable_cards.sort()
            return gainable_cards
        else:
            return list(set(list(map(lambda card: card.play, player.hand))))
    
    def _action_to_string(self,action):
        return f"{_ALL_CARDS[action-1]._action_to_string(action)} to hand from supply"

    def _apply_action(self, state, action):
        player = state.get_current_player()
        if self.card_to_gain is None:
            card = _get_card(action)
            state.supply_piles[card.name].qty -= 1
            self.card_to_gain = card
            player.hand.append(self.card_to_gain)
        else:
            card = _get_card(action)
            player.hand.remove(card)
            player.draw_pile.append(card)
            state.effect_runner.effects[player.id] = None

    def run(self, state, player):
        state.effect_runner.initiator = player.id
        state.effect_runner.add_effect(player.id, self)


class SentryEffect(Effect):
    """ Look at the top 2 cards of your deck. Trash and or discard any number of them. Put the rest back on top in any order. """

    def __init__(self):
        self.name = "Sentry"
        self.num_cards_trashed_or_discarded = 0
        self.top_two_cards = []
        self.id = 15
    
    def __eq__(self,other):
        return True

    def _legal_actions(self, state, player):
        discardable_cards = list(set([card.discard for card in self.top_two_cards]))
        trashable_cards = list(set([card.trash for card in self.top_two_cards]))
        all_cards = discardable_cards + trashable_cards + [END_PHASE_ACTION]
        all_cards.sort()
        return all_cards

    def _apply_action(self, state, action):
        player = state.get_current_player()
        card = _get_card(action)
        if action is card.trash:
            player.trash_pile.append(card)
        elif action is card.discard:
            player.discard_pile.append(card)
        else:
            state.effect_runner.effects[player.id] = None
            return 
        player.draw_pile.remove(card)
        self.top_two_cards.remove(card)
        self.num_cards_trashed_or_discarded += 1
        if self.num_cards_trashed_or_discarded is len(self.top_two_cards):
            state.effect_runner.effects[player.id] = None

    
    def run(self, state, player):
        state.effect_runner.initiator = player.id
        state.effect_runner.add_effect(player.id, self)
        # todo - what happens when we don't have 2 cards in deck?
        self.top_two_cards = player.draw_pile[0:2]


class HarbingerEffect(Effect):
    def __init__(self):
        self.name = "Harbinger"
        self.id = 16
    def __eq__(self,other):
        return True

    def _legal_actions(self, state, player):
        actions = list(set(list(map(lambda card: card.gain, player.discard_pile)))) + [END_PHASE_ACTION]
        actions.sort()
        return actions
    def _action_to_string(self,action):
        return f"{_get_card(action)._action_to_string(action)} to draw pile from discard_pile" if action is not END_PHASE_ACTION else f"End {self.name}"

    def _apply_action(self, state, action):
        player = state.get_current_player()
        if action is END_PHASE_ACTION:
            state.effect_runner.effects[player.id] = None
        else:
            card = _get_card(action)
            player.discard_pile.remove(card)
            player.draw_pile.insert(0, card)
            state.effect_runner.effects[player.id] = None

    def run(self, state, player):
        state.effect_runner.initiator = player.id
        state.effect_runner.add_effect(player.id, self)


class LibraryEffect(Effect):
    def __init__(self):
        self.id = 17
        pass

    def _legal_actions(self, state, player):
        pass

    def run(self, state, player):
        pass

class BanditOppEffect():
    def __init__(self):
        pass
    def run(self,state,player):
        pass


""" TREASURE CARDS """
COPPER = TreasureCard(1, name='Copper', cost=0, coins=1)
SILVER = TreasureCard(2, name='Silver', cost=3, coins=2)
GOLD = TreasureCard(3, name='Gold', cost=6, coins=3)
""" VICTORY CARDS """
CURSE = VictoryCard(4, name='Curse', cost=0, victory_points=-1)
DUCHY = VictoryCard(5, name='Duchy', cost=4, victory_points=5)
ESTATE = VictoryCard(6, name='Estate', cost=2, victory_points=1)
PROVINCE = VictoryCard(7, name='Province', cost=8, victory_points=8)
""" KINGDOM CARDS """
VILLAGE = ActionCard(8, name='Village', cost=3, add_actions=2, add_cards=1)
LABORATORY = ActionCard(9, name='Laboratory', cost=5, add_cards=2, add_actions=1)
FESTIVAL = ActionCard(10, name='Festival', cost=5, add_actions=2, add_buys=1, coins=2)
MARKET = ActionCard(11, name='Market', cost=5, add_actions=1, add_buys=1, coins=1, add_cards=1)
SMITHY = ActionCard(12, name="Smithy", cost=4, add_cards=3)
MILITIA = AttackCard(13, name='Militia', cost=4, coins=2, effect_list=[lambda: OpponentsDiscardDownToEffect(3)],add_cards=0)
GARDENS = VictoryCard(14, name='Gardens', cost=4, victory_points=0,vp_fn=lambda all_cards: math.floor(len(all_cards) / 10))
CHAPEL = ActionCard(15, name='Chapel', cost=2,effect_list=[lambda: TrashCardsEffect(num_cards=4,name='Chapel',optional=True)])
WITCH = AttackCard(16, name='Witch', cost=5, add_cards=2, effect_list=[lambda: OpponentsGainCardEffect(CURSE)], coins=0)
WORKSHOP = ActionCard(17, name='Workshop', cost=3, effect_list=[lambda: ChoosePileToGainEffect(4)])
BANDIT = AttackCard(18,name = 'Bandit', cost = 5, effect_list=[lambda: GainCardToDiscardPileEffect(GOLD), lambda: BanditOppEffect()], coins = 0, add_cards = 0)
REMODEL = ActionCard(19, name='Remodel', cost=4, effect_list=[lambda: TrashAndGainCostEffect(2, False)])
THRONE_ROOM = ActionCard(20, name='Throne Room', cost=4)
MONEYLENDER = ActionCard(21, name='Moneylender', cost=4,effect_list=[lambda: TrashTreasureAndGainCoinEffect(treasure_card=COPPER, n_coins=3)])
POACHER = ActionCard(22, name='Poacher', cost=4, add_cards=1, add_actions=1, coins=1,effect_list=[lambda: PoacherEffect()])
MERCHANT = ActionCard(23, name='Merchant', cost=3, add_cards=1, add_actions=1) 
CELLAR = ActionCard(24, name='Cellar', cost=2, add_actions=1, effect_list=[lambda: CellarEffect()])
MINE = ActionCard(25, name='Mine', cost=5, effect_list=[lambda: TrashTreasureAndGainTreasure(n_coins=3)])
VASSAL = ActionCard(26, name='Vassal', cost=3, coins=2, effect_list=[lambda: VassalEffect()])
COUNCIL_ROOM = ActionCard(27, name='Council Room', cost=5, add_cards=4, add_buys=1,effect_list=[lambda: OpponentsGainCardToHandEffect(num_cards=1)])
ARTISAN = ActionCard(28, name='Artisan', cost=6, effect_list=[lambda: ArtisanEffect()])
BUREAUCRAT = ActionCard(29, name='Bureaucrat', cost=4) 
SENTRY = ActionCard(30, name='Sentry', cost=5, add_cards=1, add_actions=1, effect_list=[lambda: SentryEffect()])
HARBINGER = ActionCard(31, name='Harbinger', cost=3, add_cards=1, add_actions=1, effect_list=[lambda: HarbingerEffect()])
LIBRARY = ActionCard(32, name='Library', cost=5, effect_list=[lambda: LibraryEffect()])
MOAT = ReactionCard(33, name='Moat', cost=2, add_cards=2)


def create_kingdom_supplies(): 
    return dict({
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
    })

def create_treasure_cards_supply():
    return dict({
        'Copper': SupplyPile(COPPER, 46),
        'Silver': SupplyPile(SILVER, 40),
        'Gold': SupplyPile(GOLD, 30)
    })



def create_victory_cards_supply():
    return dict({
        'Curse': SupplyPile(CURSE, 10),
        'Estate': SupplyPile(ESTATE, 8),
        'Duchy': SupplyPile(DUCHY, 8),
        'Province': SupplyPile(PROVINCE, 8)
    })



NUM_TREASURE_PILES = 3
NUM_VICTORY_PILES = 4

END_PHASE_ACTION = 166

_TREASURE_CARDS = [COPPER, SILVER, GOLD]
_TREASURE_CARDS_NAMES = list(map(lambda card: card.name, _TREASURE_CARDS))

_VICTORY_CARDS = [CURSE, ESTATE, DUCHY, PROVINCE]
_VICTORY_CARDS_NAMES = list(map(lambda card: card.name, _VICTORY_CARDS))

_ALL_CARDS = [COPPER, SILVER, GOLD, CURSE, DUCHY, ESTATE, PROVINCE, VILLAGE, LABORATORY, FESTIVAL, MARKET, SMITHY,
              MILITIA, GARDENS, CHAPEL, WITCH, WORKSHOP, BANDIT, REMODEL, THRONE_ROOM, MONEYLENDER, POACHER, MERCHANT,
              CELLAR, MINE, VASSAL, COUNCIL_ROOM, ARTISAN, BUREAUCRAT, SENTRY, HARBINGER, LIBRARY, MOAT]



def _get_card(action):
        id = (action - 1) % len(_ALL_CARDS)
        return _ALL_CARDS[id]

class TurnPhase(enumerate):
    ACTION_PHASE = 1
    TREASURE_PHASE = 2
    BUY_PHASE = 3
    END_TURN = 4

def turn_phase_to_str(turn_phase):
    if turn_phase is TurnPhase.ACTION_PHASE:
        return "Play Action card phase"
    elif turn_phase is TurnPhase.TREASURE_PHASE:
        return "Play Treasure card phase"
    elif turn_phase is TurnPhase.BUY_PHASE:
        return "Buy card phase"
    elif turn_phase is TurnPhase.END_TURN:
        return "End Turn phase"

def get_card_names(pile_nm: str ,list_of_cards: list):
   return f"{pile_nm}: {','.join([str(card) for card in list_of_cards])}" if len(list_of_cards) > 0 else f"{pile_nm}: Empty"

class Player(object):
    def __init__(self, id):
        self.id = id
        self.vp = 0
        self.draw_pile = [COPPER for _ in range(7)] + [ESTATE for _ in range(3)]
        self.discard_pile = []
        self.trash_pile = []
        self.cards_in_play = []
        self.hand = []
        self.phase = TurnPhase.TREASURE_PHASE
        self.actions = 1
        self.buys = 1
        self.coins = 0

        random.shuffle(self.draw_pile)
        self._draw_hand()

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        pieces = []
        pieces.append(f"p{self.id}:")
        pieces.append(get_card_names("\tdraw_pile",self.draw_pile))
        pieces.append(get_card_names("\thand",self.hand))
        pieces.append(get_card_names("\tdiscard_pile",self.discard_pile))
        pieces.append(get_card_names("\ttrash_pile",self.trash_pile))
        pieces.append(get_card_names("\tcards_in_play",self.cards_in_play))
        pieces.append(f"\tnum_actions: {self.actions}")
        pieces.append(f"\tnum_buys: {self.buys}")
        pieces.append(f"\tnum_coins: {self.coins}")
        return "\n".join(str(p) for p in pieces)

    def _draw_hand(self, num_cards: int = _HAND_SIZE):
        if len(self.draw_pile) < num_cards:
            self._add_discard_pile_to_draw_pile()
        self.hand += self.draw_pile[0:num_cards]
        self.draw_pile = self.draw_pile[num_cards:len(self.draw_pile)]

    def play_treasure_card_from_hand(self, card: Card):
        self.coins += card.coins
        self.cards_in_play.append(card)
        self.hand.remove(card)

    def play_action_card_from_hand(self, state, card: ActionCard):
        self.actions -= 1
        self.coins += card.coins or 0
        self.buys += card.add_buys or 0
        self.actions += card.add_actions or 0
        self.cards_in_play.append(card)
        if card.add_cards:
            self._draw_hand(card.add_cards)
        for effect in card.effect_list:
            effect().run(state, self)
        self.hand.remove(card)

    def buy_card(self, card: Card):
        self.draw_pile.append(card)
        self.coins -= card.cost
        self.buys -= 1
    
    def has_cards(self,card_names):
        avail_counts = Counter(list(map(lambda card: card.name,self.draw_pile + self.hand)))
        request_card_counts = Counter(card_names)
        for name in request_card_counts:
            if request_card_counts[name] > avail_counts[name]:
                return False
        return True

    @property
    def all_cards(self):
        return self.hand + self.draw_pile + self.discard_pile + self.cards_in_play

    @property
    def victory_points(self):
        total = self.vp
        for card in list(filter(lambda card: isinstance(card, VictoryCard), self.all_cards)):
            total += card.victory_points
            if card.vp_fn:
                total += card.vp_fn(self.all_cards)
        return total

    def load_hand(self, cards):
        self.draw_pile += self.hand
        self.hand.clear()
        for card in cards:
            if card not in self.draw_pile:
                raise Exception(f"{card.name} was not found in p{self.id}'s draw_pile")
            else:
                self.hand.append(card)
                self.draw_pile.remove(card)
        if len(self.hand) < _HAND_SIZE:
            self._draw_hand(_HAND_SIZE - len(self.hand))

    def re_init_turn(self, cards):
        self.load_hand(cards)
        self.phase = TurnPhase.ACTION_PHASE if self.has_action_cards else TurnPhase.TREASURE_PHASE

    def end_phase(self):
        if self.phase is TurnPhase.ACTION_PHASE:
            self.phase = TurnPhase.TREASURE_PHASE
        elif self.phase is TurnPhase.TREASURE_PHASE and self.coins == 0:
            self.phase = TurnPhase.END_TURN
        elif self.phase is TurnPhase.TREASURE_PHASE:
            self.phase = TurnPhase.BUY_PHASE
        elif self.phase is TurnPhase.BUY_PHASE:
            self.phase = TurnPhase.END_TURN
        return self.phase

    def _add_hand_cards_in_play_to_discard_pile(self):
        self.discard_pile += self.hand
        self.discard_pile += self.cards_in_play
        self.hand.clear()
        self.cards_in_play.clear()

    def _add_discard_pile_to_draw_pile(self):
        random.shuffle(self.discard_pile)
        self.draw_pile += self.discard_pile
        self.discard_pile.clear()

    @property
    def has_action_cards(self):
        return next((card for card in self.hand if isinstance(card, ActionCard)), None) is not None

    def end_turn(self):
        '''
        1) take all cards you have in play (both actions and treasures) and any remaining cards in your hand
        and put them in your discard pile 

        2) draw a new hand of 5 cards from your draw_pile. If draw pile has fewer than 5 cards
        first shuffle your discard_pile and put it under your deck, then draw

        3) actions, buys, coins restore to default
        '''
        # cleanup-phase
        self._add_hand_cards_in_play_to_discard_pile()
        self._draw_hand()
        self.actions = 1
        self.coins = 0
        self.buys = 1
        self.phase = TurnPhase.ACTION_PHASE if self.has_action_cards else TurnPhase.TREASURE_PHASE


class DominionGame(pyspiel.Game):
    """ A python version of Dominion."""

    def __init__(self, params=None):
        self._GAME_INFO = pyspiel.GameInfo(
            num_distinct_actions=167,
            max_chance_outcomes=0,
            num_players=params["num_players"],
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0,
            max_game_length=3
        )
        num_kingdom_cards = len(params['kingdom_cards'].split(", "))
        if num_kingdom_cards is not _NUM_KINGDOM_SUPPLY_PILES:
            raise Exception(f"Expected list of 10 kingdom cards got: {num_kingdom_cards}")
        super().__init__(_GAME_TYPE, self._GAME_INFO, params or dict())

        self._init_kingdom_supply = create_kingdom_supplies()
        self._init_treasure_supply = create_treasure_cards_supply()
        self._init_victory_supply = create_victory_cards_supply()

    def get_game_info(self):
        return self._GAME_INFO

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return DominionGameState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        all_kingdom_cards = [card for card in _ALL_CARDS if card.name in self._init_kingdom_supply.keys()]
        return DominionObserver(iig_obs_type, {"num_players": self.get_game_info().num_players,
                                               "kingdom_cards": self.get_parameters()['kingdom_cards'].split(", "),
                                               "all_kingdom_cards": sorted(all_kingdom_cards,key=lambda card: card.id)})


class EffectRunner:
    def __init__(self, num_players):
        self.effects = [None] * num_players
        self.initiator = None

    @property
    def active(self):
        return len([effect for effect in self.effects if effect is not None]) != 0

    @property
    def active_player(self):
        for i, effect in enumerate(self.effects):
            if effect is not None:
                return i
        return self.initiator
    
    @property
    def active_effect(self):
        return self.effects[self.active_player]

    def _legal_actions(self, state, player):
        return self.effects[player.id]._legal_actions(state, player)

    def add_effect(self, player, effect: Effect):
        self.effects[player] = effect

    def _apply_action(self, state, action):
        self.effects[self.active_player]._apply_action(state, action)
        return self.active_player


class DominionGameState(pyspiel.State):
    """ a python version of the Dominon state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)

        self._players = [Player(i) for i in range(game.num_players())]
        self._kingdom_card_names = game.get_parameters()['kingdom_cards'].split(", ")
        self._curr_player = 0
        self._is_terminal = False

        kingdom_piles = {key: game._init_kingdom_supply[key] for key in game._init_kingdom_supply if key in self._kingdom_card_names}

        self.supply_piles = {}
        self.supply_piles.update(game._init_treasure_supply)
        self.supply_piles.update(kingdom_piles)
        self.supply_piles.update(game._init_victory_supply)

        self.effect_runner = EffectRunner(game.num_players())
        self.default_observer = game.make_py_observer()

    @property
    def victory_points(self):
        return list(map(lambda p: p.victory_points, self._players))

    def is_terminal(self):
        no_provinces_left = self.supply_piles[PROVINCE.name].qty == 0
        three_piles_empty = len(list(filter(lambda supply: supply.qty == 0, self.supply_piles.values()))) == 3
        self._is_terminal = no_provinces_left or three_piles_empty
        return self._is_terminal

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self.effect_runner.active:
            self._curr_player = self.effect_runner.active_player
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._curr_player

    def _legal_actions(self, player_id: int) -> list:
        player = self.get_player(player_id)
        if self.effect_runner.active:
            return self.effect_runner._legal_actions(self, player)
        elif player.phase is TurnPhase.TREASURE_PHASE:
            return self.legal_treasure_cards(player)
        elif player.phase is TurnPhase.BUY_PHASE:
            return self.legal_cards_to_buy(player)
        elif player.phase is TurnPhase.ACTION_PHASE:
            return self.legal_action_cards(player)

    def legal_treasure_cards(self, player):
        """ player can play their treasure cards in exchange for coins and end current phase"""
        treasure_cards = set(
            list(map(lambda card: card.play, filter(lambda card: isinstance(card, TreasureCard), player.hand))))
        return list(treasure_cards) + [END_PHASE_ACTION]

    def legal_cards_to_buy(self, player):
        """ player can buy any card whose buy value is less than or equal to player's coins and card supply > 0; end
        current phase """
        is_valid_card_to_buy = lambda card: card.name in self.supply_piles and self.supply_piles[
            card.name].qty > 0 and self.supply_piles[card.name].card.cost <= player.coins
        all_valid_cards = list(map(lambda card: card.buy, filter(is_valid_card_to_buy, _ALL_CARDS)))
        return all_valid_cards + [END_PHASE_ACTION]

    def legal_action_cards(self, player):
        """ player can play any action card in their hand """
        is_action_card_in_hand = lambda card: isinstance(card, ActionCard) and card in player.hand
        all_action_cards_in_hand = list(
            map(lambda card: card.play, filter(is_action_card_in_hand, _ALL_CARDS)))
        return all_action_cards_in_hand + [END_PHASE_ACTION]

    def _action_to_string(self, player, action):
        if self.effect_runner.active:
            return self.effect_runner.active_effect._action_to_string(action)
        elif action is END_PHASE_ACTION:
            return "End phase"
        else:
            id = (action-1) % len(_ALL_CARDS)
            card = _ALL_CARDS[id]
            return card._action_to_string(action)

    def get_player(self, id):
        return self._players[id]

    def get_players(self) -> list:
        return self._players

    def other_players(self, player):
        return [p.id for p in self._players if p is not player]

    def get_current_player(self):
        return self.get_player(self._curr_player)

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        pieces = []
        kingdom_supply_piles = ", ".join([f"{card_nm}: {self.supply_piles[card_nm].qty}" for card_nm in self._kingdom_card_names])
        treasure_piles = ", ".join([f"{card_nm}: {self.supply_piles[card_nm].qty}" for card_nm in _TREASURE_CARDS_NAMES])
        victory_piles = ", ".join([f"{card_nm}: {self.supply_piles[card_nm].qty}" for card_nm in _VICTORY_CARDS_NAMES])
        victory_points = ", ".join([f"p{player.id}: {player.victory_points}" for player in self._players])
        players = "\n".join([str(player) for player in self._players])
        pieces.append(f"kingdom supply piles: {kingdom_supply_piles}")
        pieces.append(f"treasure supply piles: {treasure_piles}")
        pieces.append(f"victory supply piles: {victory_piles}")
        pieces.append(f"victory points: {victory_points}")
        pieces.append(f"turn phase: {turn_phase_to_str(self.get_current_player().phase)}")
        pieces.append((f"current player:{self.current_player()}"))
        pieces.append(players)             
        return "\n".join(str(p) for p in pieces)

    def play_treasure_card(self, card: TreasureCard):
        player = self.get_current_player()
        player.play_treasure_card_from_hand(card)
        all_treasure_cards_played = len(list(filter(lambda card: isinstance(card, TreasureCard), player.hand))) == 0
        if all_treasure_cards_played:
            self.play_end_phase(player)

    def play_buy_card(self, card: Card):
        player = self.get_current_player()
        if player.buys is 0:
            raise Exception(f"Player {player.id} does not have any buys")
        else:
            self.supply_piles[card.name].qty -= 1
        player.buy_card(card)
        if player.buys is 0:
            self.play_end_phase(player)

    def play_action_card(self, card: ActionCard):
        player = self.get_current_player()
        if player.actions is 0:
            raise Exception(f"Player {player.id} does not have any actions")
        player.play_action_card_from_hand(self, card)
        if player.actions is 0 or not player.has_action_cards:
            self.play_end_phase(player)

    def play_end_phase(self, player):
        uptd_phase = player.end_phase()
        if uptd_phase is TurnPhase.END_TURN:
            player.end_turn()
            self.move_to_next_player()

    def _apply_action(self, action):
        if self.is_terminal():
            raise GameFinishedException("Game is finished")
        player = self.current_player()
        _legal_actions = self._legal_actions(self.current_player())
        if action not in _legal_actions:
            action_str = lambda action: f"{action}:{self._action_to_string(self.current_player(),action)}"
            _legal_actions_str = ", ".join(list(map(action_str, _legal_actions)))
            raise Exception(f"Action {action_str(action)} not in list of legal actions - {_legal_actions_str}")
        else:
            if self.effect_runner.active:
                self._curr_player = self.effect_runner._apply_action(self, action)
            elif action is END_PHASE_ACTION:
                self.play_end_phase(self.get_player(player))
            else:
                player = self.get_current_player()
                card = _get_card(action)
                if player.phase is TurnPhase.TREASURE_PHASE:
                    self.play_treasure_card(card)
                elif player.phase is TurnPhase.BUY_PHASE:
                    self.play_buy_card(card)
                else:
                    self.play_action_card(card)

    def move_to_next_player(self):
        self._curr_player = (self._curr_player + 1) % len(self._players)
    

    def load_hand(self, card_names):
        player = self.get_current_player()
        if len(card_names) > _HAND_SIZE:
            raise Exception("List of cards to load into player's hand must be <= 5")
        elif len(player.hand) != _HAND_SIZE:
            raise Exception("load hand can only be called at start of player's turn")
        elif not player.has_cards(card_names):
            raise Exception(F"{card_names} must be available for draw. Cards available for player's hand: {list(map(lambda card: card.name,player.hand + player.draw_pile))}")
        cards = [self.supply_piles[name].card for name in card_names]
        player.re_init_turn(cards)
    
    def returns(self):
        winner_vp = max(list(map(lambda player: player.victory_points, self._players)))
        return [1 if player.victory_points is winner_vp else -1 for player in self._players]
    
    def observation_tensor(self):
        self.default_observer.set_from(self,self.current_player())
        return self.default_observer.tensor
    
    def observation_string(self):
        self.default_observer.set_from(self,self.current_player())
        return self.default_observer.string_from(self,self.current_player())
    
    def observation_dict(self):
        self.default_observer.set_from(self,self.current_player())
        return self.default_observer.dict

class DominionObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        self._kingdom_cards = params['kingdom_cards']
        self.__all_kingdom_Cards = params['all_kingdom_cards']
        num_kingdom_cards = len(self.__all_kingdom_Cards)
        """Initializes an empty observation tensor."""
        # different components of observation
        pieces = [
            ("kingdom_cards_in_play", num_kingdom_cards, (num_kingdom_cards,)),
            ("kingdom_piles", num_kingdom_cards, (num_kingdom_cards,)),
            ("treasure_piles", NUM_TREASURE_PILES, (NUM_TREASURE_PILES,)),
            ("victory_piles", NUM_VICTORY_PILES, (NUM_VICTORY_PILES,)),
            ("victory_points", params["num_players"], (params["num_players"],)),
            ('TurnPhase', 1, (1,)),
            ('actions', 1, (1,)),
            ('buys', 1, (1,)),
            ('coins', 1, (1,)),
            ('draw', len(_ALL_CARDS), (len(_ALL_CARDS),)),
            ('hand', len(_ALL_CARDS), (len(_ALL_CARDS),)),
            ('cards_in_play', len(_ALL_CARDS), (len(_ALL_CARDS),)),
            ('discard', len(_ALL_CARDS), (len(_ALL_CARDS),)),
            ('trash', len(_ALL_CARDS), (len(_ALL_CARDS),)),
            ('effect', 1, (1,))
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

    def _count_cards(self, cards):
        return np.unique(list(map(lambda card: card.name, cards)), return_counts=True)

    def _num_all_cards(self, kingdom_card_names, pile: list):
        """treasure_cards,victory_cards,kingdom_cards"""
        num_cards = dict.fromkeys(list(map(lambda card: card.name, _ALL_CARDS)), 0)
        cards, nums = self._count_cards(pile)
        for card, num in zip(cards, nums):
            num_cards[card] = num
        return list(num_cards.values())

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        idx = 0
        kingdom_piles = [state.supply_piles[card.name].qty if card.name in state._kingdom_card_names else 0 for card in self.__all_kingdom_Cards]
        victory_piles = [state.supply_piles[card_nm].qty for card_nm in _VICTORY_CARDS_NAMES]
        treasure_piles = [state.supply_piles[card_nm].qty for card_nm in _TREASURE_CARDS_NAMES]

        effect = [state.effect_runner.effects[player].id] if state.effect_runner.effects[player] is not None else [0]
        values = [
            ("kingdom_cards_in_play", [1 if card.name in self._kingdom_cards else 0 for card in self.__all_kingdom_Cards]),
            ('kingdom_piles', kingdom_piles),
            ('treasure_piles', treasure_piles),
            ('victory_piles', victory_piles),
            ('victory_points', state.victory_points),
            ('TurnPhase', [state.get_player(player).phase]),
            ('actions', [state.get_player(player).actions]),
            ('buys', [state.get_player(player).buys]),
            ('coins', [state.get_player(player).coins]),
            ('draw', self._num_all_cards(state._kingdom_card_names, state.get_player(player).draw_pile)),
            ('hand', self._num_all_cards(state._kingdom_card_names, state.get_player(player).hand)),
            ('cards_in_play', self._num_all_cards(state._kingdom_card_names, state.get_player(player).cards_in_play)),
            ('discard', self._num_all_cards(state._kingdom_card_names, state.get_player(player).discard_pile)),
            ('trash', self._num_all_cards(state._kingdom_card_names, state.get_player(player).trash_pile)),
            ('effect', effect),
        ]

        for name, value in values:
            self.dict[name] = value
            self.tensor[idx: idx + len(value)] = value
            idx += len(value)

    def _string_count_cards(self, cards):
        unique_cards, num_unique = self._count_cards(cards)
        return ", ".join([f"{card}: {qty}" for card, qty in zip(unique_cards, num_unique)])

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        pieces.append(f"p{player}: ")
        kingdom_supply_piles = ", ".join([f"{card_nm}: {state.supply_piles[card_nm].qty}" for card_nm in state._kingdom_card_names])
        treasure_piles = ", ".join([f"{card_nm}: {state.supply_piles[card_nm].qty}" for card_nm in _TREASURE_CARDS_NAMES])
        victory_piles = ", ".join([f"{card_nm}: {state.supply_piles[card_nm].qty}" for card_nm in _VICTORY_CARDS_NAMES])
        victory_points = ", ".join([f"p{player}: {vp}" for player, vp in enumerate(state.victory_points)])
        effect = str(state.effect_runner.effects[player]) if state.effect_runner.effects[player] is not None else "none"

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
        cards_in_play = self._string_count_cards(state.get_player(player).cards_in_play)
        pieces.append(f"cards in play: {cards_in_play if len(cards_in_play) > 0 else 'empty'}")
        discard_pile = self._string_count_cards(state.get_player(player).discard_pile)
        pieces.append(f"discard pile: {discard_pile if len(discard_pile) > 0 else 'empty'}")
        trash_pile = self._string_count_cards(state.get_player(player).trash_pile)
        pieces.append(f"trash pile: {trash_pile if len(trash_pile) > 0 else 'empty'}")
        pieces.append(f"effect: {effect}")
        return "\n".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, DominionGame)