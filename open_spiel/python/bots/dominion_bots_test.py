from xml import dom
from absl.testing import absltest
from absl import flags
from collections import Counter 
from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python.bots import dominion_bots 
import numpy as np
import pyspiel 
from open_spiel.python.games import dominion

FLAGS = flags.FLAGS

KINGDOM_CARDS = "Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop"
NUM_PLAYERS = 2
params = {'num_players': NUM_PLAYERS, 'kingdom_cards': KINGDOM_CARDS}


def current_player_state(state):
    return state.get_player(state._curr_player)

def player_state(state,p_id):
    return state.get_player(p_id)

def player_has_cards(player,card_nms):
    avail_counts = Counter(list(map(lambda card: card.name,player.draw_pile)))
    request_card_counts = Counter(card_nms)
    for name in request_card_counts:
        if request_card_counts[name] > avail_counts[name]:
            return False
    return True

def load_hand(state,card_nms):
    player = current_player_state(state)
    assert player.phase == dominion.TurnPhase.TREASURE_PHASE, "load_hand can only be called at start of player's turn"
    assert len(card_nms) <= 5, "load_hand can only be called witb card_names <= 5"
    player.draw_pile += player.hand
    assert player_has_cards(player,card_nms), F"{card_nms} must be available for draw. Cards available for player's hand: {list(map(lambda card: card.name, player.draw_pile))}"
    player.hand.clear()
    cards = [state.supply_piles[name].card for name in card_nms]
    for card in cards:
        player.hand.append(card)
        player.draw_pile.remove(card)
    if player.has_action_cards:
        player.phase = dominion.TurnPhase.ACTION_PHASE

class BigMoneyBot(absltest.TestCase):

    def test_is_penultimate_province(self):
        game = pyspiel.load_game('python_dom',params)
        observer = game.make_py_observer()
        state = game.new_initial_state()
        player_0_deck_order = [dominion.COPPER.id] * 3 + [dominion.ESTATE.id] * 2 + [dominion.COPPER.id] * 4 + [dominion.ESTATE.id]
        player_1_deck_order = [dominion.COPPER.id] * 3 + [dominion.ESTATE.id] * 2 + [dominion.COPPER.id] * 4 + [dominion.ESTATE.id]
        for a in player_0_deck_order:
            state.apply_action(a)
        for a in player_1_deck_order:
            state.apply_action(a)
        bot = dominion_bots.BigMoneyBot(observer=observer,player_id=0)
        action = bot.step(state)
        self.assertFalse(bot._is_penultimate_province())
        state.supply_piles[dominion.PROVINCE.name].qty = 1
        action = bot.step(state)
        self.assertTrue(bot._is_penultimate_province()) 

    def test_buys_silver_with_3_coins(self):
        game = pyspiel.load_game('python_dom',params)
        observer = game.make_py_observer()
        bots = [
            dominion_bots.BigMoneyBot(observer=observer,player_id=0),
            dominion_bots.BigMoneyBot(observer=observer,player_id=1),
        ]
        state = game.new_initial_state()
        player_0_deck_order = [dominion.COPPER.id] * 3 + [dominion.ESTATE.id] * 2 + [dominion.COPPER.id] * 4 + [dominion.ESTATE.id]
        player_1_deck_order = [dominion.COPPER.id] * 3 + [dominion.ESTATE.id] * 2 + [dominion.COPPER.id] * 4 + [dominion.ESTATE.id]
        for a in player_0_deck_order:
            state.apply_action(a)
        for a in player_1_deck_order:
            state.apply_action(a)
        
        actual_actions = []

        while state.current_player() == 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actual_actions.append(action)

        expected_actions = [dominion.COPPER.play] * 3  + [dominion.END_PHASE_ACTION] + [dominion.SILVER.buy] + [dominion.END_PHASE_ACTION]
        np.testing.assert_array_equal(actual_actions,expected_actions)
    
    def test_buys_silver_with_4_coins(self):
        game = pyspiel.load_game('python_dom',params)
        observer = game.make_py_observer()
        bots = [
            dominion_bots.BigMoneyBot(observer=observer,player_id=0),
            dominion_bots.BigMoneyBot(observer=observer,player_id=1),
        ]
        state = game.new_initial_state()
        player_0_deck_order = [dominion.COPPER.id] * 4 + [dominion.ESTATE.id] * 1 + [dominion.COPPER.id] * 3 + [dominion.ESTATE.id] * 2
        player_1_deck_order = [dominion.COPPER.id] * 4 + [dominion.ESTATE.id] * 1 + [dominion.COPPER.id] * 3 + [dominion.ESTATE.id] * 2
        for a in player_0_deck_order:
            state.apply_action(a)
        for a in player_1_deck_order:
            state.apply_action(a)
        
        actual_actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actual_actions.append(action)

        expected_actions = [dominion.COPPER.play] * 4 + [dominion.END_PHASE_ACTION] + [dominion.SILVER.buy] + [dominion.END_PHASE_ACTION]
        np.testing.assert_array_equal(actual_actions,expected_actions)

    def test_buys_silver_with_5_coins(self):
        game = pyspiel.load_game('python_dom',params)
        observer = game.make_py_observer()
        bots = [
            dominion_bots.BigMoneyBot(observer=observer,player_id=0),
            dominion_bots.BigMoneyBot(observer=observer,player_id=1),
        ]
        state = game.new_initial_state()
        player_0_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        player_1_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        for a in player_0_deck_order:
            state.apply_action(a)
        for a in player_1_deck_order:
            state.apply_action(a)
        
        actual_actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actual_actions.append(action)


        expected_actions = [dominion.COPPER.play] * 5 + [dominion.END_PHASE_ACTION,dominion.SILVER.buy,dominion.END_PHASE_ACTION]
        np.testing.assert_array_equal(expected_actions,actual_actions)
    
    def test_buy_gold_with_6_coins(self):
        game = pyspiel.load_game('python_dom',params)
        observer = game.make_py_observer()
        bots = [
            dominion_bots.BigMoneyBot(observer=observer,player_id=0),
            dominion_bots.BigMoneyBot(observer=observer,player_id=1),
        ]
        state = game.new_initial_state()
        player_0_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        player_1_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        for a in player_0_deck_order:
            state.apply_action(a)
        for a in player_1_deck_order:
            state.apply_action(a)
        
        current_player_state(state).draw_pile.append(dominion.SILVER)
        
        load_hand(state,['Silver','Copper','Copper','Copper','Copper'])
        
        actual_actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actual_actions.append(action)
        expected_actions = [dominion.COPPER.play] * 4 + [dominion.SILVER.play,dominion.END_PHASE_ACTION,dominion.GOLD.buy,dominion.END_PHASE_ACTION] 
        np.testing.assert_array_equal(expected_actions,actual_actions)

    def test_buys_gold_with_7_coins(self):
        game = pyspiel.load_game('python_dom',params)
        observer = game.make_py_observer()
        bots = [
            dominion_bots.BigMoneyBot(observer=observer,player_id=0),
            dominion_bots.BigMoneyBot(observer=observer,player_id=1),
        ]
        state = game.new_initial_state()
        player_0_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        player_1_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        for a in player_0_deck_order:
            state.apply_action(a)
        for a in player_1_deck_order:
            state.apply_action(a)
    
        current_player_state(state).draw_pile += [dominion.SILVER] * 2
        load_hand(state,['Copper','Copper','Copper','Silver','Silver'])
        actual_actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actual_actions.append(action)

        expected_actions = [dominion.COPPER.play] * 3 + [dominion.SILVER.play] * 2 + [dominion.END_PHASE_ACTION,dominion.GOLD.buy,dominion.END_PHASE_ACTION]
        np.testing.assert_array_equal(expected_actions,actual_actions)

    def test_buys_province_with_8_coins(self):
        game = pyspiel.load_game('python_dom',params)
        observer = game.make_py_observer()
        bots = [
            dominion_bots.BigMoneyBot(observer=observer,player_id=0),
            dominion_bots.BigMoneyBot(observer=observer,player_id=1),
        ]
        state = game.new_initial_state()
        player_0_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        player_1_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        for a in player_0_deck_order:
            state.apply_action(a)
        for a in player_1_deck_order:
            state.apply_action(a)
    
        current_player_state(state).draw_pile += [dominion.SILVER] * 3
        load_hand(state,['Copper','Copper','Silver','Silver','Silver'])
        actual_actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actual_actions.append(action)

        expected_actions = [dominion.COPPER.play] * 2 + [dominion.SILVER.play] * 3 + [dominion.END_PHASE_ACTION,dominion.PROVINCE.buy,dominion.END_PHASE_ACTION]
        np.testing.assert_array_equal(expected_actions,actual_actions)
    
    def test_buys_duchy_if_has8coins_duchydancing_penultimate_province(self):
        game = pyspiel.load_game('python_dom',params)
        observer = game.make_py_observer()
        bots = [
            dominion_bots.BigMoneyBot(observer=observer,player_id=0,duchy_dancing=True),
            dominion_bots.BigMoneyBot(observer=observer,player_id=1),
        ]
        state = game.new_initial_state()
        player_0_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        player_1_deck_order = [dominion.COPPER.id] * 7 + [dominion.ESTATE.id] * 3
        for a in player_0_deck_order:
            state.apply_action(a)
        for a in player_1_deck_order:
            state.apply_action(a)
    
        current_player_state(state).draw_pile += [dominion.SILVER] * 3
        state.supply_piles[dominion.PROVINCE.name].qty = 1
        load_hand(state,['Copper','Copper','Silver','Silver','Silver'])
        actual_actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actual_actions.append(action)

        expected_actions = [dominion.COPPER.play] * 2 + [dominion.SILVER.play] * 3 + [dominion.END_PHASE_ACTION,dominion.DUCHY.buy,dominion.END_PHASE_ACTION]
        np.testing.assert_array_equal(expected_actions,actual_actions)
if __name__ == "__main__":
    absltest.main()

