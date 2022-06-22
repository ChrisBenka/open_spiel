import numpy as np
from operator import itemgetter
from collections import Counter 
import pyspiel

from open_spiel.python.games.dominion_effects import *

PLAY_ID = iter(range(1,34))
BUY_ID = iter(range(34,67))
DISCARD_ID = iter(range(67,100))
TRASH_ID = iter(range(100,133))
GAIN_ID = iter(range(133,166))

def get_card_names(pile_nm: str ,list_of_cards: list):
   return f"{pile_nm}: {','.join([str(card) for card in list_of_cards])}" if len(list_of_cards) > 0 else f"{pile_nm}: Empty"

class SupplyPile:
    def __init__(self, card, qty):
        self.card = card
        self.qty = qty

class Card(object):
    def __init__(self, id: int, name: str, cost: int):
        self.name = name
        self.cost = cost
        self.id = id
        self.__set_actions__()
    
    def __hash__(self):
        return hash(self.id)

    def __set_actions__(self):
        self.play = next(PLAY_ID)
        self.buy = next(BUY_ID)
        self.discard = next(DISCARD_ID)
        self.trash = next(TRASH_ID)
        self.gain = next(GAIN_ID)

        self.action_strs = {self.play: "Play", self.buy: "Buy", self.discard: "Discard", self.trash: "Trash", self.gain: "Gain"}

    def __eq__(self, other):
        return self.name == other.name
    
    def __str__(self):
        return self.name

    def action_to_string(self,action):
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
CHAPEL = ActionCard(15, name='Chapel', cost=2,effect_list=[lambda: TrashCardsEffect(num_cards=4)])
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
LIBRARY = ActionCard(32, name='Library', cost=5, effect_list=[lambda: LibraryEffect()]) #todo
MOAT = ReactionCard(33, name='Moat', cost=2, add_cards=2)


_ALL_CARDS = [COPPER, SILVER, GOLD, CURSE, DUCHY, ESTATE, PROVINCE, VILLAGE, LABORATORY, FESTIVAL, MARKET, SMITHY,
              MILITIA, GARDENS, CHAPEL, WITCH, WORKSHOP, BANDIT, REMODEL, THRONE_ROOM, MONEYLENDER, POACHER, MERCHANT,
              CELLAR, MINE, VASSAL, COUNCIL_ROOM, ARTISAN, BUREAUCRAT, SENTRY, HARBINGER, LIBRARY, MOAT]
_NUM_PLAYERS = 2

_NUM_KINGDOM_SUPPLY_PILES = 10
_NUM_TREASURE_PILES = 3
_NUM_VICTORY_PILES = 4

_HAND_SIZE = 5

_DEFAULT_PARAMS = {
    'num_players': _NUM_PLAYERS,
    'verbose': True,
    'kingdom_cards': None
}

def _get_card(action):
        id = (action - 1) % len(_ALL_CARDS)
        return _ALL_CARDS[id]

class TurnPhase(enumerate):
    ACTION_PHASE = 1
    TREASURE_PHASE = 2
    BUY_PHASE = 3
    END_TURN = 4

END_PHASE_ACTION = 166
_TREASURE_CARDS = [COPPER, SILVER, GOLD]
_TREASURE_CARDS_NAMES = list(map(lambda card: card.name, _TREASURE_CARDS))

_VICTORY_CARDS = [CURSE, ESTATE, DUCHY, PROVINCE]
_VICTORY_CARDS_NAMES = list(map(lambda card: card.name, _VICTORY_CARDS))
_KINGDOM_CARDS = [VILLAGE,LABORATORY,FESTIVAL,MARKET,SMITHY,MILITIA,GARDENS,CHAPEL,WITCH,WORKSHOP,BANDIT,
REMODEL,THRONE_ROOM,MONEYLENDER,POACHER,MERCHANT,CELLAR,MINE,VASSAL,COUNCIL_ROOM,ARTISAN,BUREAUCRAT,SENTRY,HARBINGER,LIBRARY,MOAT]
_KINGDOM_CARDS_NAMES = list(map(lambda card: card.name, _KINGDOM_CARDS))

_GAME_TYPE = pyspiel.GameType(
    short_name="python_dom",
    long_name="Python Dom",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
    default_loadable=False,
    parameter_specification=_DEFAULT_PARAMS
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions = 5 * len(_ALL_CARDS) + 1,
    max_chance_outcomes=len(_ALL_CARDS),
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=3
)

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
_INITIAL_SUPPLY = 10


'''
2 chance events: 
- inital kingdom card supply            X
- order of the initial draw pile        
- if len(draw_pile) < cards_to_be_Draw:
    order in which cards are added back to the draw pile. 
'''

class Player(object):
    def __init__(self, id):
        self.id = id
        self.vp = 0
        self.draw_pile = []
        self.discard_pile = []
        self.trash_pile = []
        self.cards_in_play = []
        self.hand = []
        self.phase = TurnPhase.TREASURE_PHASE
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.required_cards = None
    
    @property
    def all_cards(self):
        return self.draw_pile + self.discard_pile + self.trash_pile + self.cards_in_play + self.hand

    def add_to_draw_pile(self,card):
        self.draw_pile.append(card)
    
    def draw_hand(self,state,num_cards=_HAND_SIZE):
        state.add_discard_pile_to_draw_pile = len(self.draw_pile) < num_cards
        if state.add_discard_pile_to_draw_pile:
            self.required_cards = num_cards
        else:
            self.hand += self.draw_pile[0:num_cards]
            self.draw_pile = self.draw_pile[num_cards:len(self.draw_pile)]
    
    @property
    def has_treasure_cards_in_hand(self):
        return len(list(filter(lambda card: isinstance(card,TreasureCard),self.hand))) > 0

    @property
    def has_action_cards(self):
        return next((card for card in self.hand if isinstance(card, ActionCard)), None) is not None

    @property
    def victory_points(self):
        total = self.vp
        for card in list(filter(lambda card: isinstance(card, VictoryCard), self.all_cards)):
            total += card.victory_points
            if card.vp_fn:
                total += card.vp_fn(self.all_cards)
        return total

    def play_treasure_card(self,card):
        self.hand.remove(card)
        self.coins += card.coins
        self.cards_in_play.append(card)

    def buy_card(self, card):
        self.draw_pile.append(card)
        self.coins -= card.cost
        self.buys -= 1

    def play_action_card(self, state, card):
        self.hand.remove(card)
        self.actions -= 1
        self.coins += card.coins or 0
        self.buys += card.add_buys or 0
        self.actions += card.add_actions or 0
        self.cards_in_play.append(card)
        # for effect in card.effect_list:
        #     effect().run(state, self)
        if card.add_cards:
            self.draw_hand(state,card.add_cards)
       
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

    def add_to_draw_pile_from_discard_pile(self,card):
        self.draw_pile.append(card)
        self.discard_pile.remove(card)

    def end_turn(self,state):
        self._add_hand_cards_in_play_to_discard_pile()
        self.draw_hand(state)
        self.actions = 1
        self.coins = 0
        self.buys = 1
        self.phase = TurnPhase.ACTION_PHASE if self.has_action_cards else TurnPhase.TREASURE_PHASE

class DominionGame(pyspiel.Game):
    def __init__(self, params = None):
        if params['kingdom_cards'] is not None:
            DominionGame.validate_kingdom_cards(params['kingdom_cards'])
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

        self._init_kingdom_supply = create_kingdom_supplies()
        self._init_treasure_supply = create_treasure_cards_supply()
        self._init_victory_supply = create_victory_cards_supply()
        self._provided_kingdom_cards = set()
        if params['kingdom_cards'] is not None:
            self._provided_kingdom_cards = params['kingdom_cards'].split(",")
    
    @staticmethod
    def validate_kingdom_cards(cards):
        cards = set(cards.split(","))
        if len(cards) !=  _NUM_KINGDOM_SUPPLY_PILES:
            raise Exception("Expected list of 10 unique kingdom cards separated by a comma")
        for card in cards:
            if card not in _KINGDOM_CARDS_NAMES:
                avail_kingdom_cards = ", ".join(_KINGDOM_CARDS_NAMES)
                raise Exception(f"{card} is not an available kingdom card. Available kingdom cards: \n {avail_kingdom_cards}")

    def new_initial_state(self):
        return DominionGameState(self)

class DominionGameState(pyspiel.State):
    def __init__(self,game):
        super().__init__(game)

        self._curr_player = 0
        self.is_terminal = False
        self.supply_piles = {}
        self.supply_piles.update(game._init_treasure_supply)
        self.supply_piles.update(game._init_victory_supply)
        self._kingdom_supply_dict = game._init_kingdom_supply
        self._kingdom_supply = {}

        self._players = [Player(i) for i in range(game.num_players())]
        self._num_players = game.num_players()

        if len(game._provided_kingdom_cards) != 0:
            self.populate_kingdom_supply_piles(game._provided_kingdom_cards)
    
    def get_player(self,player_id):
        return self._players[player_id]

    def populate_kingdom_supply_piles(self,provided_kingdom_cards):
        for card in provided_kingdom_cards:
            self._kingdom_supply[card] =  self._kingdom_supply_dict[card]
            self.supply_piles[card] = self._kingdom_supply_dict[card]
    
    def game_finished(self):
        no_provinces_left = self.supply_piles[PROVINCE.name].qty == 0
        three_piles_empty = len(list(filter(lambda supply: supply.qty == 0, self.supply_piles.values()))) == 3
        return no_provinces_left or three_piles_empty

    def chance_outcomes(self):
        assert self.is_chance_node()
        if len(self._kingdom_supply.keys()) < _NUM_KINGDOM_SUPPLY_PILES:
            outcomes = set(_KINGDOM_CARDS_NAMES).difference(set(self._kingdom_supply.keys()))
            p = 1.0/len(outcomes)
            return sorted([(self._kingdom_supply_dict[outcome].card.id,p) for outcome in outcomes])
        elif not self.each_player_received_init_supply():
            num_coppers = len(list(filter(lambda card: card.name == COPPER.name, self._players[self._curr_player].all_cards)))
            num_estates = len(list(filter(lambda card: card.name == ESTATE.name, self._players[self._curr_player].all_cards)))
            total = _INITIAL_SUPPLY - (num_coppers + num_estates)
            copper_p = (7 - num_coppers) / total
            estate_p = (3 - num_estates) / total
            return sorted([(COPPER.id,copper_p),(ESTATE.id,estate_p)])
        elif self.add_discard_pile_to_draw_pile:
            discard_pile = self._players[self._curr_player].discard_pile
            unique_cards_in_discard_pile = list(set(discard_pile))
            counts = Counter(discard_pile)
            return sorted([(card.id,counts[card]/len(discard_pile)) for card in unique_cards_in_discard_pile])

    def each_player_received_init_supply(self):
        val = True
        for p in reversed(range(self._num_players)):
            received_init_supply =  len(self._players[p].all_cards) >= _INITIAL_SUPPLY
            val = val and received_init_supply
        return val

    def current_player(self):
        if len(self._kingdom_supply.keys()) < _NUM_KINGDOM_SUPPLY_PILES:
            return pyspiel.PlayerId.CHANCE
        elif not self.each_player_received_init_supply():
            self._curr_player = next(player for player in reversed(self._players) if len(player.all_cards) < _INITIAL_SUPPLY).id
            return pyspiel.PlayerId.CHANCE
        elif self.add_discard_pile_to_draw_pile:
            return pyspiel.PlayerId.CHANCE
        return self._curr_player

    def add_to_kingdom_supply(self,card):
        self._kingdom_supply[card.name] = self._kingdom_supply_dict[card.name]
        self.supply_piles[card.name] = self._kingdom_supply_dict[card.name]
 
    def _apply_chance_action(self,action):
        if len(self._kingdom_supply.keys()) < _NUM_KINGDOM_SUPPLY_PILES:
            card = _get_card(action)
            self.add_to_kingdom_supply(card)
        elif not self.each_player_received_init_supply():
            card = _get_card(action)
            self._players[self._curr_player].add_to_draw_pile(card)
            if self.each_player_received_init_supply():
                for p in self._players:
                    p.draw_hand(self)
        elif self.add_discard_pile_to_draw_pile:
            card = _get_card(action)
            self._players[self._curr_player].add_to_draw_pile_from_discard_pile(card)
            self.add_discard_pile_to_draw_pile = len(self._players[self._curr_player].discard_pile) != 0
            if not self.add_discard_pile_to_draw_pile:
                player = self._players[self._curr_player]
                player.draw_hand(self,player.required_cards)

    def _move_to_next_player(self):
        self._curr_player = (self._curr_player + 1) % len(self._players)
    
    def _play_treasure_card(self,card):
        player = self._players[self._curr_player]
        player.play_treasure_card(card)

    def _buy_card(self,card):
        player = self._players[self._curr_player]
        self.supply_piles[card.name].qty -=1
        player.buy_card(card)

    def _play_action_card(self,card):
        player = self._players[self._curr_player]
        player.play_action_card(self, card)

    def _apply_end_phase_action(self):
        player = self._players[self._curr_player]
        updtd_phase = player.end_phase()
        if updtd_phase is TurnPhase.END_TURN:
            player.end_turn(self)
            self._move_to_next_player() 
    
    def _action_to_string(self,p_id,action):
        if action is END_PHASE_ACTION:
            return "End Phase"
        else:
            return _get_card(action).action_to_string(action)

    def _apply_action(self,action):
        if self.is_chance_node():
            self._apply_chance_action(action)
        else:
            _legal_actions = self._legal_actions(self._curr_player)
            if action not in _legal_actions:
                action_str = lambda action: f"{action}:{self._action_to_string(self.current_player(),action)}"
                _legal_actions_str = ", ".join(list(map(action_str, _legal_actions)))
                raise Exception(f"Action {action_str(action)} not in list of legal actions - {_legal_actions_str}")
            elif action is END_PHASE_ACTION:
                self._apply_end_phase_action()
            else:
                player = self._players[self._curr_player]
                card = _get_card(action)
                if player.phase is TurnPhase.TREASURE_PHASE:
                    self._play_treasure_card(card)
                elif player.phase is TurnPhase.BUY_PHASE:
                    self._buy_card(card)
                elif player.phase is TurnPhase.ACTION_PHASE:
                    self._play_action_card(card)
        self._is_terminal = self.game_finished()
            
    def _legal_action_cards(self, p_id):
        """ player can play any action card in their hand """
        player = self._players[p_id]
        if player.actions is 0:
            return [END_PHASE_ACTION]
        is_action_card_in_hand = lambda card: isinstance(card, ActionCard) and card in player.hand
        all_action_cards_in_hand = list(
            map(lambda card: card.play, filter(is_action_card_in_hand, _ALL_CARDS)))
        return sorted(all_action_cards_in_hand + [END_PHASE_ACTION])

    def _legal_cards_to_buy(self,p_id):
        player = self._players[p_id]
        if player.buys is 0:
            return [END_PHASE_ACTION]
        is_valid_card_to_buy = lambda card: card.name in self.supply_piles and self.supply_piles[
            card.name].qty > 0 and self.supply_piles[card.name].card.cost <= player.coins
        all_valid_cards = list(map(lambda card: card.buy, filter(is_valid_card_to_buy, _ALL_CARDS)))
        return sorted(all_valid_cards + [END_PHASE_ACTION])

    def _legal_treasure_cards(self, p_id):
        """ player can play their treasure cards in exchange for coins and end current phase"""
        player = self._players[p_id]
        treasure_cards = set(
            list(map(lambda card: card.play, filter(lambda card: isinstance(card, TreasureCard), player.hand))))
        return sorted(list(treasure_cards) + [END_PHASE_ACTION])
   
    def _legal_actions(self, p_id):
        """Returns a list of legal actions, sorted in ascending order."""
        assert p_id >= 0
        player = self._players[p_id]
        if player.phase is TurnPhase.TREASURE_PHASE:
            return self._legal_treasure_cards(p_id)
        elif player.phase is TurnPhase.BUY_PHASE:
            return self._legal_cards_to_buy(p_id)
        elif player.phase is TurnPhase.ACTION_PHASE:
            return self._legal_action_cards(p_id)
        

        
pyspiel.register_game(_GAME_TYPE, DominionGame)

'''
3 chance nodes
1. initial cards if none are provided
2. order in which cards are assigned to each player's draw pile on load of game
3. order in which cards are added to a player's draw pile as they run out of cards. 
'''