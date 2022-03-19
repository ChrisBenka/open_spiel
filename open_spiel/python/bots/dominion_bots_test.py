from absl.testing import absltest
from absl import flags
from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python.bots import dominion_bots 
import numpy as np
import pyspiel 
from open_spiel.python.games import dominion

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "python_dominion", "Name of the game")
flags.DEFINE_integer("num_players", 2, "Number of players")
flags.DEFINE_string("kingdom_cards","Village, Laboratory, Festival, Market, Militia, Gardens, Chapel, Throne Room, Moneylender, Poacher", "names of 10 kingdom cards for gameplay")



class BigMoneyBot(absltest.TestCase):

    def test_buys_silver_with_3_coins(self):
        game = pyspiel.load_game(FLAGS.game, {"num_players": FLAGS.num_players,"kingdom_cards": FLAGS.kingdom_cards})
        bots = [
            dominion_bots.BigMoneyBot(0),
            dominion_bots.BigMoneyBot(1)
        ]
        state = game.new_initial_state()
        #mock hand
        state.load_hand(['Copper','Copper','Copper','Estate','Estate'])
        actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actions.append(action)

        expected_actions = [dominion.COPPER.play] * 3 + [dominion.SILVER.buy]
        np.testing.assert_array_equal(expected_actions,actions)
    
    def test_buys_silver_with_4_coins(self):
        game = pyspiel.load_game(FLAGS.game, {"num_players": FLAGS.num_players,"kingdom_cards": FLAGS.kingdom_cards})
        bots = [
            dominion_bots.BigMoneyBot(0),
            dominion_bots.BigMoneyBot(1)
        ]
        state = game.new_initial_state()
        #mock hand
        state.load_hand(['Copper','Copper','Copper','Copper','Estate'])
        actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actions.append(action)

        expected_actions = [dominion.COPPER.play] * 4 + [dominion.SILVER.buy]
        np.testing.assert_array_equal(expected_actions,actions)

    def test_buys_silver_with_5_coins(self):
        game = pyspiel.load_game(FLAGS.game, {"num_players": FLAGS.num_players,"kingdom_cards": FLAGS.kingdom_cards})
        bots = [
            dominion_bots.BigMoneyBot(0),
            dominion_bots.BigMoneyBot(1)
        ]
        state = game.new_initial_state()
        #mock hand
        state.load_hand(['Copper','Copper','Copper','Copper','Copper'])
        actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actions.append(action)

        expected_actions = [dominion.COPPER.play] * 5 + [dominion.SILVER.buy]
        np.testing.assert_array_equal(expected_actions,actions)
    
    def test_buys_gold_with_6_coins(self):
        game = pyspiel.load_game(FLAGS.game, {"num_players": FLAGS.num_players,"kingdom_cards": FLAGS.kingdom_cards})
        bots = [
            dominion_bots.BigMoneyBot(0),
            dominion_bots.BigMoneyBot(1)
        ]
        state = game.new_initial_state()
        #mock draw_pile and hand
        state.get_current_player().draw_pile += [dominion.SILVER]
        state.load_hand(['Copper','Copper','Copper','Copper','Silver'])
        actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actions.append(action)

        expected_actions = [dominion.COPPER.play] * 4 + [dominion.SILVER.play] + [dominion.GOLD.buy]
        np.testing.assert_array_equal(expected_actions,actions)

    def test_buys_gold_with_7_coins(self):
        game = pyspiel.load_game(FLAGS.game, {"num_players": FLAGS.num_players,"kingdom_cards": FLAGS.kingdom_cards})
        bots = [
            dominion_bots.BigMoneyBot(0),
            dominion_bots.BigMoneyBot(1)
        ]
        state = game.new_initial_state()
        #mock draw_pile and hand
        state.get_current_player().draw_pile += [dominion.SILVER] * 2
        state.load_hand(['Copper','Copper','Copper','Silver','Silver'])
        actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actions.append(action)

        expected_actions = [dominion.COPPER.play] * 3 + [dominion.SILVER.play] * 2 + [dominion.GOLD.buy]
        np.testing.assert_array_equal(expected_actions,actions)

    def test_buys_province_with_8_coins(self):
        game = pyspiel.load_game(FLAGS.game, {"num_players": FLAGS.num_players,"kingdom_cards": FLAGS.kingdom_cards})
        bots = [
            dominion_bots.BigMoneyBot(0),
            dominion_bots.BigMoneyBot(1)
        ]
        state = game.new_initial_state()
        #mock draw_pile and hand
        state.get_current_player().draw_pile += [dominion.GOLD] * 2
        state.load_hand(['Copper','Copper','Estate','Gold','Gold'])
        actions = []
        while state.current_player() is 0:
            action = bots[0].step(state)
            state.apply_action(action)
            actions.append(action)

        expected_actions = [dominion.COPPER.play] * 2 + [dominion.GOLD.play] * 2 + [dominion.PROVINCE.buy]
        np.testing.assert_array_equal(expected_actions,actions)

if __name__ == "__main__":
    absltest.main()

