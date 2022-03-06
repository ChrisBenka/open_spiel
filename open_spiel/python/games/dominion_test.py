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

# Lint as python3
"""Tests for Python Dominion."""

import numpy as np
import random
from absl.testing import absltest
from open_spiel.python.games import dominion


class DominionTestState(absltest.TestCase):

    def test_can_create_and_state(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertIsNotNone(state)

    def test_state_has_supply_piles_victory_points(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertIsNotNone(state.kingdom_piles)
        self.assertIsNotNone(state.victory_piles)
        self.assertIsNotNone(state.treasure_piles)
        self.assertIsNotNone(state.victory_points)
        self.assertEqual(len(state.victory_points),dominion._DEFAULT_PARAMS['num_players'])

    def test_each_player_starts_with_7coppers_3_estates_in_draw_piles(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        num_cards = lambda card_name, cards: len(list(filter(lambda card: card.name == card_name, cards)))
        for p in range(dominion._DEFAULT_PARAMS['num_players']):
            player = state.get_player(p)
            self.assertEqual(num_cards(dominion.COPPER.name,player.hand + player.draw_pile),7)
            self.assertEqual(num_cards(dominion.ESTATE.name,player.hand + player.draw_pile),3)

    def test_first_player_draws_5cards_from_discard_pile_to_start_game(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertEqual(len(state.get_player(0).hand), dominion._HAND_SIZE)

    def test_isTerminal_When0ProvincesLeft(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.victory_piles[dominion.PROVINCE.name].qty = 0
        self.assertTrue(state.is_terminal())

    def test_isTerminal_When3SupplyPilesEmpty(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.kingdom_piles[dominion.VILLAGE.name].qty = 0
        state.treasure_piles[dominion.GOLD.name].qty = 0
        state.kingdom_piles[dominion.BUREAUCRAT.name].qty = 0
        self.assertTrue(state.is_terminal())

    def test_load_hand(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.load_hand(['Copper','Copper','Estate','Copper','Copper'])
        player = state.get_player(state.current_player())
        self.assertEqual(player.hand,[dominion.COPPER,dominion.COPPER,dominion.ESTATE,dominion.COPPER,dominion.COPPER])

        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.load_hand(['Copper','Copper','Estate'])
        player = state.get_player(state.current_player())
        self.assertEqual(player.hand[0:3],[dominion.COPPER,dominion.COPPER,dominion.ESTATE])
        self.assertEqual(len(player.hand),dominion._HAND_SIZE)

# class DominionObserverTest(absltest.TestCase):

    # def test_dominion_observation(self):
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     observation = game.make_py_observer()
    #     state = game.new_initial_state()

    #     state.load_hand(['Copper','Copper','Copper','Estate','Estate'])

    #     observation.set_from(state, player=0)

    #     self.assertEqual(list(observation.dict),
    #                      ["kingdom_cards_in_play","kingdom_piles", "treasure_piles", "victory_piles", "victory_points", "TurnPhase", "actions",
    #                       "buys",
    #                       "coins", "draw", "hand", "cards_in_play","discard", "trash"])

    #     np.testing.assert_equal(observation.tensor.shape, (230,))
    #     np.testing.assert_array_equal(observation.dict["kingdom_cards_in_play"],[1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,0, 0, 1, 1])
    #     np.testing.assert_array_equal(observation.dict["treasure_piles"], [46, 40, 30])
    #     np.testing.assert_array_equal(observation.dict["victory_piles"], [10, 8, 8, 8])
    #     np.testing.assert_array_equal(observation.dict["victory_points"], [3, 3])
    #     np.testing.assert_array_equal(observation.dict["coins"], [0])
    #     np.testing.assert_array_equal(observation.dict["buys"], [1])
    #     np.testing.assert_array_equal(observation.dict["actions"], [1])
    #     np.testing.assert_array_equal(observation.dict["draw"], [4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #     np.testing.assert_array_equal(observation.dict["hand"], [3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #     np.testing.assert_equal(observation.dict["discard"], np.zeros(33))
    #     np.testing.assert_equal(observation.dict["trash"], np.zeros(33))
    #     np.testing.assert_equal((observation.dict["cards_in_play"]), np.zeros(33))

    #     np.testing.assert_equal(np.sum(observation.dict["draw"]) + np.sum(observation.dict["hand"]), 10)

    # def test_dominion_observation_str(self):
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     observation = game.make_py_observer()
    #     state = game.new_initial_state()
    #     state.load_hand(['Copper','Copper','Copper','Estate','Estate'])
    #     obs_str = 'p0: \nkingdom supply piles: Village: 10, Market: 10, Smithy: 10, Militia: 10, Witch: 10, Mine: 10, Council Room: 10, Bureaucrat: 10, Library: 10, Moat: 10\ntreasure supply piles: Copper: 46, Silver: 40, Gold: 30\nvictory supply piles: Curse: 10, Estate: 8, Duchy: 8, Province: 8\nvictory points: p0: 3, p1: 3\nTurn Phase: 2\nactions: 1\nbuys: 1\ncoin: 0\ndraw pile: Copper: 4, Estate: 1\nhand: Copper: 3, Estate: 2\ncards in play: empty\ndiscard pile: empty\ntrash pile: empty'
    #     self.assertEqual(observation.string_from(state, player=0),obs_str)

    # def test_treasure_phase_obs(self):
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     observation = game.make_py_observer()
    #     state = game.new_initial_state()

    #     state.load_hand(['Copper','Copper','Copper','Estate','Estate'])
    #     play_copper = 1
    #     state.apply_action(play_copper)
    #     state.apply_action(play_copper)
    #     observation.set_from(state, player=0)
    #     self.assertEqual(observation.dict['cards_in_play'],[2] + [0] * 32)


class DominionPlayerTurnTest(absltest.TestCase):

    # def test_players_firstTurn_starts_0coins_1actions_1buy(self):
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     observation = game.make_py_observer()
    #     state = game.new_initial_state()
    #     observation.set_from(state, state.current_player())
    #     self.assertEqual(observation.dict['buys'][0], 1)
    #     self.assertEqual(observation.dict['actions'][0], 1)
    #     self.assertEqual(observation.dict['coins'][0], 0)
    #     self.assertEqual(observation.dict['TurnPhase'][0], dominion.TurnPhase.TREASURE_PHASE)

    # def test_firstTurn_StartsTreasurePhase_InitiallegalAction_TreasureCardsAndEndPhase(self):
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     observation = game.make_py_observer()
    #     state = game.new_initial_state()
    #     state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
    #     actions = state.legal_actions()
    #     copper_action = 1
    #     observation.set_from(state, player=0)
    #     # player starts with 0 action cards so we skip to Treasure_Phase
    #     self.assertEqual(observation.dict['TurnPhase'], [dominion.TurnPhase.TREASURE_PHASE])
    #     self.assertEqual(actions, [copper_action, dominion.END_PHASE_ACTION]);

    def test_BuyPhase_ActionsAreThoseCardsThatCanBePurchased_AndEndPhase(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()
        state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
        buy_copper = dominion.COPPER.id
        end_phase = dominion.END_PHASE_ACTION
        state.apply_action(buy_copper)
        state.apply_action(buy_copper)
        state.apply_action(buy_copper)
        state.apply_action(end_phase)

        actions = state.legal_actions()
        valid_actions = [dominion.COPPER.id, dominion.SILVER.id,dominion.CURSE.id,dominion.ESTATE.id, dominion.VILLAGE.id, dominion.MOAT.id,dominion.END_PHASE_ACTION]
        self.assertEqual(actions, valid_actions)

    def test_Action_to_String(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()
        state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
        self.assertEqual(state.action_to_string(dominion.COPPER.id), "Play Copper")

        player = state.get_player(0)
        player.coins = 3
        player.phase = dominion.TurnPhase.BUY_PHASE

        self.assertEqual(state.action_to_string(dominion.COPPER.id), "Buy and Gain Copper")

    # def test_ApplyAction_ExceptionRaised_WhenGameDone(self):
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     state = game.new_initial_state()
    #     state.victory_piles[dominion.PROVINCE.name].qty = 0
    #     try: 
    #         state.apply_action(dominion.COPPER.id)
    #     except Exception as e:
    #         self.assertEqual(str(e), "Game is finished")

    def test_ApplyAction_ExceptionRaised_WhenActionNotLegal(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
        try:
            state.apply_action(dominion.SILVER.id)
        except Exception as e:
            self.assertEqual(str(e), "Action 2:Play Silver not in list of legal actions - 1:Play Copper, 34:End phase")

    def test_canPlay_TreasurePhase_AutoEnd(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        initial_state = game.new_initial_state()
        initial_state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
        # play all coppers in hand
        curr_player = initial_state.get_player(initial_state.current_player())
        for _ in range(4):
            initial_state.apply_action(dominion.COPPER.id)
        updtd_num_coppers_in_hand = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))
        updtd_num_coppers_in_play = len(list(filter(lambda card: card.name is 'Copper', curr_player.cards_in_play)))
        self.assertEqual(curr_player.coins, 4)
        self.assertEqual(updtd_num_coppers_in_hand, 0)
        self.assertEqual(updtd_num_coppers_in_play,4)
        self.assertEqual(len(curr_player.hand), dominion._HAND_SIZE - 4)

        # player plays all treasure cards, game will move onto BUY_PHASE automatically
        self.assertEqual(curr_player.phase, dominion.TurnPhase.BUY_PHASE)

    def test_canPlay_TreasurePhase_EndPhase(self):
        # #play all coppers in hand - 1 + END_PHASE
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
        curr_player = state.get_player(state.current_player())
        for _ in range(3):
            state.apply_action(dominion.COPPER.id)
        state.apply_action(dominion.END_PHASE_ACTION)
        updtd_num_coppers_in_hand = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))
        updtd_num_coppers_in_play = len(list(filter(lambda card: card.name is 'Copper', curr_player.cards_in_play)))
        self.assertEqual(curr_player.coins,3)
        self.assertEqual(updtd_num_coppers_in_hand, 1)
        self.assertEqual(updtd_num_coppers_in_play, 3)
        self.assertEqual(curr_player.phase, dominion.TurnPhase.BUY_PHASE)

    # def test_can_buy_treasure_card(self):
    #     # play all coppers in hand and buy a silver card
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     state = game.new_initial_state()
    #     state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
    #     curr_player = state.get_player(state.current_player())
    #     num_coppers = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))

    #     for _ in range(num_coppers):
    #         state.apply_action(dominion.COPPER.id)
    #     state.apply_action(dominion.SILVER.id)
    #     self.assertIn(dominion.SILVER.name, list(map(lambda card: card.name, curr_player.draw_pile)))
    #     self.assertEqual(state.treasure_piles[dominion.SILVER.name].qty, 39)
    #     self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.END_TURN)
    #     self.assertEqual(state.get_player(0).victory_points, 3)
    #     self.assertEqual(state.current_player(), 1)

    # def test_can_buy_victory_card(self):
    #     # play all coppers in hand and buy a victory card
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     state = game.new_initial_state()
    #     state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
    #     curr_player = state.get_player(state.current_player())
    #     for _ in range(4):
    #         state.apply_action(dominion.COPPER.id)
    #     state.apply_action(dominion.ESTATE.id)
    #     estate = dominion.ESTATE.name

    #     self.assertIn(dominion.ESTATE.name, list(map(lambda card: card.name, curr_player.draw_pile)))
    #     self.assertEqual(state.victory_piles[dominion.ESTATE.name].qty, 7)
    #     self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.END_TURN)
    #     self.assertEqual(state.get_player(0).victory_points, 4)
    #     self.assertEqual(state.current_player(), 1)

    # def test_can_buy_kingdom_card(self):
    #     # play all coppers in hand and buy a kingdom_card
    #     game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
    #     state = game.new_initial_state()
    #     state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
    #     curr_player = state.get_player(state.current_player())
    #     for _ in range(4):
    #         state.apply_action(dominion.COPPER.id)

    #     state.apply_action(dominion.MOAT.id)
    #     self.assertIn(dominion.MOAT.name, list(map(lambda card: card.name, curr_player.draw_pile)))
    #     self.assertEqual(state.kingdom_piles[dominion.MOAT.name].qty, 9)
    #     self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.END_TURN)

    def test_clean_up_phase(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
        curr_player = state.get_player(0)
        #skip treasure + buy phase
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.END_PHASE_ACTION)

        self.assertEqual(curr_player.actions, 1)
        self.assertEqual(curr_player.buys, 1)
        self.assertEqual(curr_player.coins, 0)
        # prior hand is moved to discard pile
        self.assertEqual(5, len(curr_player.discard_pile))
        # player draws next hand from draw pile
        self.assertEqual(5, len(curr_player.hand))
        self.assertEqual(0, len(curr_player.draw_pile))

class DominionKingdomCardEffects(absltest.TestCase):
    def test_village(self):
        """ Playing village adds 1 card, 2 actions """
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.load_hand(['Copper', 'Copper', 'Copper', 'Copper', 'Estate'])
        curr_player = state.get_player(state.current_player())
        for _ in range(4):
            state.apply_action(dominion.COPPER.id)
        # buy Village
        state.apply_action(dominion.VILLAGE.id)
        # next player skips buy phase and moves back to first player
        state.apply_action(dominion.END_PHASE_ACTION)
        # back to inital player ; play village
        state.load_hand(['Village'])
        state.apply_action(dominion.VILLAGE.id)
        self.assertEqual(state.get_player(0).actions, 2)
        # player wil not have any action cards left, move on to TREASURE_PHASE
        self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.TREASURE_PHASE)
        self.assertEqual(len(state.get_player(0).hand), 5)
    
    def test_laboratory(self):
        """add 2 cards ; add 1 action"""
        kingdom_cards = "Moat, Village, Laboratory, Smithy, Militia, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #mock draw_pile to contain at least 5 coins to purchase Laboratory
        curr_player.draw_pile = [dominion.GOLD] * 3 + [dominion.COPPER] * 4 + [dominion.ESTATE] * 3
        state.load_hand(['Gold','Gold','Copper','Gold','Estate'])
        
        #play all golds + end Phase
        for _ in range(3):
            state.apply_action(dominion.GOLD.id)
        state.apply_action(dominion.END_PHASE_ACTION)
        #buy laboratory
        state.apply_action(dominion.LABORATORY.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play Laboratory
        state.load_hand([dominion.LABORATORY.name])
        state.apply_action(dominion.LABORATORY.id)

        self.assertEqual(state.get_player(0).actions, 1)
        # player wil not have any action cards left, move on to TREASURE_PHASE
        self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.TREASURE_PHASE)
        self.assertEqual(len(state.get_player(0).hand), 6)

    def test_festival(self):
        """add 2 actions ; 1 buys ; 2 coins"""
        kingdom_cards = "Moat, Village, Festival, Smithy, Militia, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #mock draw_pile to contain at least 5 coins to purchase Festival
        curr_player.draw_pile = [dominion.GOLD] * 3 + [dominion.COPPER] * 4 + [dominion.ESTATE] * 3
        state.load_hand(['Gold','Gold','Copper','Gold','Estate'])
        
        #play all golds + end Phase
        for _ in range(3):
            state.apply_action(dominion.GOLD.id)
        state.apply_action(dominion.END_PHASE_ACTION)
        #buy Festival
        state.apply_action(dominion.FESTIVAL.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play Festival
        state.load_hand([dominion.FESTIVAL.name])
        state.apply_action(dominion.FESTIVAL.id)

        self.assertEqual(state.get_player(0).coins, 2)
        self.assertEqual(state.get_player(0).buys, 2)
        self.assertEqual(state.get_player(0).actions, 2)
        # player wil not have any action cards left, move on to TREASURE_PHASE
        self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.TREASURE_PHASE)
    
    def test_market(self):
        """add 1 actoin, 1 buy, 1 coin, 1 card"""
        kingdom_cards = "Moat, Village, Festival, Smithy, Militia, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #mock draw_pile to contain at least 5 coins to purchase Market
        curr_player.draw_pile = [dominion.GOLD] * 3 + [dominion.COPPER] * 4 + [dominion.ESTATE] * 3
        state.load_hand(['Gold','Gold','Copper','Gold','Estate'])
        
        #play all golds + end Phase
        for _ in range(3):
            state.apply_action(dominion.GOLD.id)
        state.apply_action(dominion.END_PHASE_ACTION)
        #buy Market
        state.apply_action(dominion.MARKET.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play Market
        state.load_hand([dominion.MARKET.name])
        state.apply_action(dominion.MARKET.id)

        self.assertEqual(state.get_player(0).coins, 1)
        self.assertEqual(state.get_player(0).buys, 2)
        self.assertEqual(state.get_player(0).actions, 1)
        self.assertEqual(len(state.get_player(0).hand),5)
        self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.TREASURE_PHASE)
    
    def test_smithy(self):
        """ add 3 cards """ 
        kingdom_cards = "Moat, Village, Festival, Smithy, Militia, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #mock draw_pile to contain at least 4 coins to purchase Smithy
        curr_player.draw_pile = [dominion.GOLD] * 3 + [dominion.COPPER] * 4 + [dominion.ESTATE] * 3
        state.load_hand(['Gold','Gold','Copper','Gold','Estate'])
        
        #play all golds + end Phase
        for _ in range(3):
            state.apply_action(dominion.GOLD.id)
        state.apply_action(dominion.END_PHASE_ACTION)
        #buy Smithy
        state.apply_action(dominion.SMITHY.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play Smithy
        state.load_hand([dominion.SMITHY.name])
        state.apply_action(dominion.SMITHY.id)

        self.assertEqual(len(state.get_player(0).hand),7)
        self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.TREASURE_PHASE)
    
    def test_militia(self):
        """add 2 coins, runs OpponentsDiscardDownToEffect which causes opponents to draw down to 3 cards in their hands (cards put into their respective discard piles) """

        kingdom_cards = "Moat, Village, Festival, Smithy, Militia, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #mock draw_pile to contain at least 4 coins to purchase Militia
        curr_player.draw_pile = [dominion.GOLD] * 3 + [dominion.COPPER] * 4 + [dominion.ESTATE] * 3
        state.load_hand(['Gold','Gold','Copper','Gold','Estate'])
        
        #play all golds + end phase
        for _ in range(3):
            state.apply_action(dominion.GOLD.id)
        state.apply_action(dominion.END_PHASE_ACTION)
        #buy Militia
        state.apply_action(dominion.MILITIA.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play Militia
        state.load_hand([dominion.MILITIA.name])
        state.apply_action(dominion.MILITIA.id)

        #assert OpponentsdrawDownToThreeCards is running
        self.assertTrue(state.effect_runner.active)
        drawDownToThreeCards = dominion.DiscardDownToEffect(3)
        for p in state.other_players(state.get_player(0)):
            self.assertEqual(state.effect_runner.effects[p],drawDownToThreeCards)
        #assert player 1 is active (needs to attend to effect)
        self.assertEqual(state.current_player(),1)
        
        #player attends to effect:
        num_cards_in_discard_pile = len(state.get_player(1).discard_pile)
        while state.current_player() is 1:
            card_to_discard = random.choice(state.legal_actions())
            self.assertTrue(card_to_discard in list(map(lambda card: card.id,state.get_current_player().hand)))
            state.apply_action(card_to_discard)
        self.assertEqual(len(state.get_player(1).hand),drawDownToThreeCards.num_cards_downto)
        self.assertEqual(len(state.get_player(1).discard_pile),num_cards_in_discard_pile+2)
        
        self.assertEqual(state.current_player(),0)
        self.assertEqual(state.get_current_player().phase,dominion.TurnPhase.TREASURE_PHASE)
        self.assertEqual(state.get_current_player().coins,2)

    def test_gardens(self):
        kingdom_cards = "Moat, Village, Gardens, Smithy, Militia, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())
        
        #mock draw pile
        curr_player.draw_pile.append(dominion.VILLAGE)
        curr_player.draw_pile.append(dominion.VILLAGE)
        curr_player.draw_pile.append(dominion.VILLAGE)
        curr_player.draw_pile.append(dominion.VILLAGE)
        curr_player.draw_pile.append(dominion.VILLAGE)

        curr_player.draw_pile.append(dominion.GARDENS)

        self.assertEqual(curr_player.victory_points,4)
    
    def test_chapel(self):
        """Player can trash up to 4 cards from hand"""

        kingdom_cards = "Moat, Village, Festival, Smithy, Chapel, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 2 coins to buy a chapel
        state.load_hand(['Copper','Copper','Copper','Copper','Estate'])
        
        for _ in range(4):
            state.apply_action(dominion.COPPER.id)
        #buy chapel
        state.apply_action(dominion.CHAPEL.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play Chapel
        state.load_hand([dominion.CHAPEL.name])
        state.apply_action(dominion.CHAPEL.id)

        #assert trashCardsEffect is running
        self.assertTrue(state.effect_runner.active)
        trashCardsEffect = dominion.TrashCardsEffect(4,optional=True)
        self.assertEqual(state.effect_runner.effects[0],trashCardsEffect)
        self.assertEqual(state.current_player(),0)
        self.assertEqual(state.effect_runner.effects[1],None)

        # trash 2 cards and move on:
        for _ in range(2):
            trash_to_discard = random.choice(state.legal_actions()[:-1])
            self.assertTrue(trash_to_discard in list(map(lambda card: card.id,state.get_current_player().hand)))
            state.apply_action(trash_to_discard)
        state.apply_action(dominion.END_PHASE_ACTION)
        self.assertEqual(state.effect_runner.active,False)
        self.assertEqual(len(state.get_current_player().trash_pile),2)
        self.assertEqual(state.get_current_player().phase,dominion.TurnPhase.TREASURE_PHASE)
     

    # def test_witch_opponent_immune(self):
    #     """Since opponent immune to attack opponent will not gain a curse,but player to gain 2 cards to hand"""

    #     kingdom_cards = "Moat, Village, Festival, Smithy, Chapel, Witch, Library, Market, Mine, Council Room"
    #     game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
    #     game = dominion.DominionGame(game_params)
    #     state = game.new_initial_state()
    #     curr_player = state.get_player(state.current_player())

    #     #must have 5 coins to buy a WITCH
    #     state.load_hand(['Copper','Copper','Copper','Copper','Copper'])
        
    #     for _ in range(5):
    #         state.apply_action(dominion.COPPER.id)
    #     #buy witch
    #     state.apply_action(dominion.WITCH.id)
    #     # skip next player's turn
    #     state.apply_action(dominion.END_PHASE_ACTION)
    #     # play Witch
    #     state.load_hand([dominion.WITCH.name])
    #     state.apply_action(dominion.WITCH.id)

    #     #assert opponents have gained a curse (causes a loss of 1 VP)
    #     for p in state.other_players(state.get_player(0)):
    #         self.assertIn(dominion.CURSE,state.get_player(p).discard_pile)
    #         self.assertEqual(state.get_player(p).victory_points,2)
    #     self.assertEqual(len(state.get_current_player().hand),5)

    #     def test_witch(self):
    #     """causes opponents to gain a curse card and player to gain 2 cards to hand"""

    #     kingdom_cards = "Moat, Village, Festival, Smithy, Chapel, Witch, Library, Market, Mine, Council Room"
    #     game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
    #     game = dominion.DominionGame(game_params)
    #     state = game.new_initial_state()
    #     curr_player = state.get_player(state.current_player())

    #     #must have 5 coins to buy a WITCH
    #     state.load_hand(['Copper','Copper','Copper','Copper','Copper'])
        
    #     for _ in range(5):
    #         state.apply_action(dominion.COPPER.id)
    #     #buy witch
    #     state.apply_action(dominion.WITCH.id)
    #     # skip next player's turn
    #     state.apply_action(dominion.END_PHASE_ACTION)
    #     # play Witch
    #     state.load_hand([dominion.WITCH.name])
    #     state.apply_action(dominion.WITCH.id)

    #     #assert opponents have gained a curse (causes a loss of 1 VP)
    #     for p in state.other_players(state.get_player(0)):
    #         self.assertIn(dominion.CURSE,state.get_player(p).discard_pile)
    #         self.assertEqual(state.get_player(p).victory_points,2)
    #     self.assertEqual(len(state.get_current_player().hand),5)

    def test_workshop(self):
        """player gains a card from any pile costing less than 4 coins to their discard pile"""

        kingdom_cards = "Moat, Village, Festival, Smithy, Workshop, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 3 coins to buy a Workshop
        state.load_hand(['Copper','Copper','Copper','Copper','Copper'])
        
        for _ in range(5):
            state.apply_action(dominion.COPPER.id)
        #buy workshop
        state.apply_action(dominion.WORKSHOP.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play workshop
        state.load_hand([dominion.WORKSHOP.name])
        state.apply_action(dominion.WORKSHOP.id)

        gainEffect = dominion.ChoosePileToGainEffect(4)

        self.assertTrue(state.effect_runner.active)
        self.assertEqual(state.effect_runner.effects[0],gainEffect)
        self.assertEqual(state.current_player(),0)

        cards_less_than_equal_4_coins = list(map(lambda card: card.id,filter(lambda card: card.cost <= 4 and card.name in state._all_supply_piles,dominion._ALL_CARDS)))

        legal_actions = state.legal_actions()

        self.assertEqual(cards_less_than_equal_4_coins,legal_actions)

        state.apply_action(random.choice(legal_actions))

        self.assertFalse(state.effect_runner.active)
        self.assertEqual(state.current_player(),0)
        self.assertEqual(state.get_current_player().phase,dominion.TurnPhase.TREASURE_PHASE)

    def test_bandit(self):
        """player gains a gold card to discard pile"""

        kingdom_cards = "Moat, Village, Festival, Smithy, Bandit, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 5 coins to buy a bandit
        state.load_hand(['Copper','Copper','Copper','Copper','Copper'])
        
        for _ in range(5):
            state.apply_action(dominion.COPPER.id)
        #buy bandit
        state.apply_action(dominion.BANDIT.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play bandit
        state.load_hand([dominion.BANDIT.name])
        state.apply_action(dominion.BANDIT.id)

        self.assertEqual(state.current_player(),0)
        self.assertIn(dominion.GOLD,state.get_current_player().discard_pile)
        self.assertEqual(state.get_current_player().phase,dominion.TurnPhase.TREASURE_PHASE)

    def test_remodel(self):
        """player trashes a card and gains a card whose cost is <= cost of trashed_card + 2"""

        kingdom_cards = "Moat, Village, Festival, Smithy, Remodel, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 4 coins to buy a remodel
        state.load_hand(['Copper','Copper','Copper','Copper','Copper'])
        
        for _ in range(5):
            state.apply_action(dominion.COPPER.id)
        #buy remodel
        state.apply_action(dominion.REMODEL.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play remodel
        state.load_hand([dominion.REMODEL.name, dominion.ESTATE.name])
        state.apply_action(dominion.REMODEL.id)

        self.assertEqual(state.current_player(),0)
        self.assertTrue(state.effect_runner.active)

        effect = dominion.TrashAndGainCostEffect(2,False)

        self.assertEqual(state.effect_runner.effects[0],effect)
        
        cards_in_hand = list(set(list(map(lambda card: card.id,state.get_current_player().hand))))

        self.assertEqual(state.legal_actions(),cards_in_hand)

        #trash estate in hand
        state.apply_action(dominion.ESTATE.id)
        self.assertEqual(len(state.get_current_player().hand),3)
        self.assertEqual(len(state.get_current_player().trash_pile),1)
        self.assertIn(dominion.ESTATE,state.get_current_player().trash_pile)
        self.assertTrue(state.effect_runner.active)
        is_valid_card_to_gain = lambda pile: pile.qty > 0 and pile.card.cost <= dominion.ESTATE.cost + 2
        valid_gainable_cards = filter(is_valid_card_to_gain,state._all_supply_piles.values())
        valid_gainable_cards = list(map(lambda pile: pile.card.id, valid_gainable_cards))
        valid_gainable_cards.sort()
        self.assertEqual(valid_gainable_cards,state.legal_actions())
        state.apply_action(dominion.SILVER.id)
        self.assertIn(dominion.SILVER,state.get_current_player().discard_pile)
        self.assertFalse(state.effect_runner.active)

        self.assertEqual(state.get_current_player().phase,dominion.TurnPhase.TREASURE_PHASE)

    def test_moneylender_copper_in_hand_trash_copper(self):
        """player optionally trashes a treasure card in their hand. If they do player gains n_coins"""

        kingdom_cards = "Moat, Village, Festival, Smithy, Moneylender, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 4 coins to buy a moneylender
        state.load_hand(['Copper','Copper','Copper','Copper','Estate'])
        
        for _ in range(4):
            state.apply_action(dominion.COPPER.id)
        #buy moneylender
        state.apply_action(dominion.MONEYLENDER.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play moneylender
        state.load_hand([dominion.MONEYLENDER.name,dominion.COPPER.name])
        state.apply_action(dominion.MONEYLENDER.id)

        self.assertEqual(state.current_player(),0)

        effect = dominion.TrashTreasureAndGainCoinEffect(dominion.COPPER,3)

        self.assertTrue(state.effect_runner.active)
        self.assertEqual(state.effect_runner.effects[0],effect)
        
        self.assertEqual(state.legal_actions(),[dominion.COPPER.id,dominion.END_PHASE_ACTION])

        #trash copper in hand
        state.apply_action(dominion.COPPER.id)


        self.assertEqual(len(state.get_current_player().hand),3)
        self.assertEqual(len(state.get_current_player().trash_pile),1)

        self.assertIn(dominion.COPPER,state.get_current_player().trash_pile)
        self.assertFalse(state.effect_runner.active)
        self.assertEqual(state.get_current_player().coins,3)
        self.assertEqual(state.get_current_player().phase,dominion.TurnPhase.TREASURE_PHASE)

    def test_moneylender_copper_in_hand_no_trash(self):
        """player optionally trashes a treasure card in their hand. If they do player gains n_coins"""

        kingdom_cards = "Moat, Village, Festival, Smithy, Moneylender, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 4 coins to buy a moneylender
        state.load_hand(['Copper','Copper','Copper','Copper','Estate'])
        
        for _ in range(4):
            state.apply_action(dominion.COPPER.id)
        #buy moneylender
        state.apply_action(dominion.MONEYLENDER.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play moneylender
        state.load_hand([dominion.MONEYLENDER.name,dominion.COPPER.name])
        state.apply_action(dominion.MONEYLENDER.id)

        self.assertEqual(state.current_player(),0)

        effect = dominion.TrashTreasureAndGainCoinEffect(dominion.COPPER,3)

        self.assertTrue(state.effect_runner.active)
        self.assertEqual(state.effect_runner.effects[0],effect)
        
        self.assertEqual(state.legal_actions(),[dominion.COPPER.id,dominion.END_PHASE_ACTION])

        #trash copper in hand
        state.apply_action(dominion.END_PHASE_ACTION)


        self.assertEqual(len(state.get_current_player().hand),4)
        self.assertEqual(len(state.get_current_player().trash_pile),0)

        self.assertNotIn(dominion.COPPER,state.get_current_player().trash_pile)
        self.assertFalse(state.effect_runner.active)
        self.assertEqual(state.get_current_player().coins,0)
        self.assertEqual(state.get_current_player().phase,dominion.TurnPhase.TREASURE_PHASE)

    def test_moneylender_copper_not_in_hand(self):
        """player optionally trashes a treasure card in their hand. If they do player gains n_coins"""

        kingdom_cards = "Moat, Village, Festival, Smithy, Moneylender, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 4 coins to buy a moneylender
        state.load_hand(['Copper','Copper','Copper','Copper','Estate'])
        
        for _ in range(4):
            state.apply_action(dominion.COPPER.id)
        #buy moneylender
        state.apply_action(dominion.MONEYLENDER.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play moneylender
        state.get_current_player().draw_pile += [dominion.ESTATE] * 4
        state.load_hand([dominion.MONEYLENDER.name,dominion.ESTATE.name,dominion.ESTATE.name,dominion.ESTATE.name,dominion.ESTATE.name])
        state.apply_action(dominion.MONEYLENDER.id)

        effect = dominion.TrashTreasureAndGainCoinEffect(dominion.COPPER,3)

        self.assertFalse(state.effect_runner.active)
        self.assertEqual(state.effect_runner.effects[0],None)
        self.assertEqual(len(state.get_current_player().hand),4)
        self.assertEqual(len(state.get_current_player().trash_pile),0)
        self.assertEqual(state.get_current_player().coins,0)
        self.assertEqual(state.get_current_player().phase,dominion.TurnPhase.TREASURE_PHASE)
    def test_poacher_no_empty_supply_pile(self):
        """
        player gains a card , an action, a coin, and discards a card per empty supply pile"""
        """
        You draw your one card before discarding.
        If there are no empty piles, you do not discard. If there is one empty pile, you discard one card; if there are two empty piles, you discard two cards, and so on. This includes all Supply piles, including Curses, Coppers, Poacher itself, etc.
        If you do not have enough cards to discard, just discard the rest of your hand. 
        """
        kingdom_cards = "Moat, Village, Festival, Smithy, Poacher, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 4 coins to buy a poacher
        state.load_hand(['Copper','Copper','Copper','Copper','Estate'])
        
        for _ in range(4):
            state.apply_action(dominion.COPPER.id)
        #buy poacher
        state.apply_action(dominion.POACHER.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play poacher
        state.load_hand([dominion.POACHER.name])
        state.apply_action(dominion.POACHER.id)

        self.assertFalse(state.effect_runner.active)
        self.assertEqual(state.effect_runner.effects[0],None)
        self.assertEqual(len(state.get_current_player().hand),5)
        self.assertEqual(state.get_current_player().coins,1)
        self.assertEqual(state.get_current_player().actions,1)

    def test_cellar(self):
        """
        player gains 1 action, discards n cards, and draws n cards
        """
        kingdom_cards = "Moat, Village, Festival, Smithy, Cellar, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 4 coins to buy a poacher
        state.load_hand(['Copper','Copper','Copper','Copper','Estate'])
        
        for _ in range(4):
            state.apply_action(dominion.COPPER.id)
        #buy cellar
        state.apply_action(dominion.CELLAR.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play cellar
        state.load_hand([dominion.CELLAR.name])
        state.apply_action(dominion.CELLAR.id)
        effect = dominion.CellarEffect()
        self.assertTrue(state.effect_runner.active)
        self.assertEqual(state.get_current_player().actions,1)

        # discard 2 cards
        num_cards_in_discard = len(state.get_current_player().discard_pile)
        for _ in range(2):
            card_to_discard = random.choice(state.legal_actions()[:-1])
            state.apply_action(card_to_discard)
        state.apply_action(dominion.END_PHASE_ACTION)
        self.assertEqual(len(state.get_current_player().hand),4)
        self.assertFalse(state.effect_runner.active)
    
    def test_mine(self):
        kingdom_cards = "Moat, Village, Festival, Smithy, Cellar, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 5 coins to buy a mine
        state.load_hand(['Copper','Copper','Copper','Copper','Copper'])
        
        for _ in range(5):
            state.apply_action(dominion.COPPER.id)
        #buy mine
        state.apply_action(dominion.MINE.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play mine
        state.load_hand([dominion.MINE.name,dominion.COPPER.name])
        state.apply_action(dominion.MINE.id)
        effect = dominion.TrashTreasureAndGainTreasure(n_coins = 3)
        self.assertTrue(state.effect_runner.active)
        self.assertTrue(state.effect_runner.effects[0],effect)

        self.assertEqual([dominion.COPPER.id,dominion.END_PHASE_ACTION],state.legal_actions())
        #trash_copper
        state.apply_action(dominion.COPPER.id)
        self.assertEqual([dominion.COPPER.id,dominion.SILVER.id],state.legal_actions())
        state.apply_action(dominion.SILVER.id)
        self.assertIn(dominion.SILVER,state.get_current_player().hand)
        self.assertFalse(state.effect_runner.active)
    def test_vassal_no_action_cards(self):
        kingdom_cards = "Moat, Village, Festival, Smithy, Vassal, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 3 coins to buy a vassal
        state.load_hand(['Copper','Copper','Copper','Estate','Estate'])
        
        for _ in range(3):
            state.apply_action(dominion.COPPER.id)
        #buy vassal
        state.apply_action(dominion.VASSAL.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play vassal
        state.load_hand([dominion.VASSAL.name])
        state.apply_action(dominion.VASSAL.id)
        self.assertFalse(state.effect_runner.active)
        self.assertEqual(state.get_current_player().coins,2)
    
    def test_vassal_play_action_card(self):
        kingdom_cards = "Moat, Village, Festival, Smithy, Vassal, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 3 coins to buy a vassal
        state.load_hand(['Copper','Copper','Copper','Estate','Estate'])
        
        for _ in range(3):
            state.apply_action(dominion.COPPER.id)
        #buy vassal
        state.apply_action(dominion.VASSAL.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play vassal
        state.load_hand([dominion.VASSAL.name])
        state.get_current_player().draw_pile.insert(0,dominion.VILLAGE)
        state.apply_action(dominion.VASSAL.id)
        self.assertTrue(state.effect_runner.active)
        effect = dominion.VassalEffect()
        self.assertEqual(state.effect_runner.effects[0],effect)
        self.assertEqual(state.legal_actions(),[dominion.VILLAGE.id,dominion.END_PHASE_ACTION])
        #play village
        state.apply_action(dominion.VILLAGE.id)
        self.assertFalse(state.effect_runner.active)
        # assert village effects
        self.assertEqual(state.get_current_player().actions,3)
        self.assertEqual(len(state.get_current_player().hand),5)
    def test_vassal_do_not_play_action_card(self):
        kingdom_cards = "Moat, Village, Festival, Smithy, Vassal, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 3 coins to buy a vassal
        state.load_hand(['Copper','Copper','Copper','Estate','Estate'])
        
        for _ in range(3):
            state.apply_action(dominion.COPPER.id)
        #buy vassal
        state.apply_action(dominion.VASSAL.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play vassal
        state.load_hand([dominion.VASSAL.name])
        state.get_current_player().draw_pile.insert(0,dominion.VILLAGE)
        state.apply_action(dominion.VASSAL.id)
        self.assertTrue(state.effect_runner.active)
        effect = dominion.VassalEffect()
        self.assertEqual(state.effect_runner.effects[0],effect)
        self.assertEqual(state.legal_actions(),[dominion.VILLAGE.id,dominion.END_PHASE_ACTION])
        #play village
        state.apply_action(dominion.END_PHASE_ACTION)
        self.assertIn(dominion.VILLAGE,state.get_current_player().discard_pile)
        self.assertFalse(state.effect_runner.active)
    def test_council_room(self):
        kingdom_cards = "Moat, Village, Festival, Smithy, Vassal, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #must have 5 coins to buy a council room
        state.load_hand(['Copper','Copper','Copper','Copper','Copper'])
        
        for _ in range(5):
            state.apply_action(dominion.COPPER.id)
        #buy COUNCIL_ROOM
        state.apply_action(dominion.COUNCIL_ROOM.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play COUNCIL_ROOM
        state.load_hand([dominion.COUNCIL_ROOM.name])
        state.apply_action(dominion.COUNCIL_ROOM.id)

        #each other play gains a card to hand
        for p in state.other_players(state.get_current_player()):
            self.assertEqual(len(state.get_player(p).hand),6)

        self.assertFalse(state.effect_runner.active)
        self.assertEqual(len(state.get_current_player().hand),5)
        self.assertEqual(state.get_current_player().buys,2)
    
    def test_artisan(self):
        kingdom_cards = "Moat, Village, Festival, Smithy, Artisan, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        curr_player.draw_pile += [dominion.GOLD,dominion.GOLD]
        #must have 6 coins to buy an artisan
        state.load_hand(['Gold','Gold','Estate','Estate','Copper'])
        
        for _ in range(2):
            state.apply_action(dominion.GOLD.id)
        state.apply_action(dominion.END_PHASE_ACTION)
        #buy ARTISAN
        state.apply_action(dominion.ARTISAN.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play ARTISAN
        state.load_hand([dominion.ARTISAN.name,dominion.COPPER.name])
        state.apply_action(dominion.ARTISAN.id)

        self.assertTrue(state.effect_runner.active)
        cards_cost_up_to_5 = [1, 2, 4, 5, 6, 8, 10, 11, 12, 16, 25, 27, 32, 33]
        self.assertEqual(state.legal_actions(),cards_cost_up_to_5) 

        #choose to gain a silver
        state.apply_action(dominion.SILVER.id)
        self.assertIn(dominion.SILVER,state.get_current_player().hand)
        cards_in_hand = list(set(list(map(lambda card: card.id, state.get_current_player().hand))))
        self.assertEqual(state.legal_actions(),cards_in_hand)

        state.apply_action(dominion.COPPER.id)
        self.assertFalse(state.effect_runner.active)



class DominionPoacherEffect(absltest.TestCase):
    def test_poacher_1_empty_supply_pile(self):
        """
        player gains a card , an action, a coin, and discards a card per empty supply pile"""
        """
        You draw your one card before discarding.
        If there are no empty piles, you do not discard. If there is one empty pile, you discard one card; if there are two empty piles, you discard two cards, and so on. This includes all Supply piles, including Curses, Coppers, Poacher itself, etc.
        If you do not have enough cards to discard, just discard the rest of your hand. 
        """
        kingdom_cards = "Moat, Village, Festival, Smithy, Poacher, Witch, Library, Market, Mine, Council Room"
        game_params = {"num_players": 2, "kingdom_cards": kingdom_cards}
        game = dominion.DominionGame(game_params)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())

        #mock 1 supply pile empty
        state._all_supply_piles[dominion.COPPER.name].qty = 0

        #must have 4 coins to buy a poacher
        state.load_hand(['Copper','Copper','Copper','Copper','Estate'])
        
        for _ in range(4):
            state.apply_action(dominion.COPPER.id)
        #buy poacher
        state.apply_action(dominion.POACHER.id)
        # skip next player's turn
        state.apply_action(dominion.END_PHASE_ACTION)
        # play poacher
        state.load_hand([dominion.POACHER.name,dominion.ESTATE.name])
        state.apply_action(dominion.POACHER.id)

        effect = dominion.PoacherEffect()

        self.assertTrue(state.effect_runner.active)
        self.assertEqual(state.effect_runner.effects[0],effect)
        self.assertEqual(len(state.get_current_player().hand),5)
        self.assertEqual(state.get_current_player().coins,1)
        self.assertEqual(state.get_current_player().actions,1)

        state.apply_action(dominion.ESTATE.id)

        self.assertIn(dominion.ESTATE,state.get_current_player().discard_pile)
        self.assertTrue(state.effect_runner.active)



if __name__ == "__main__":
    absltest.main()
