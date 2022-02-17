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
from absl.testing import absltest
from open_spiel.python.games import dominion


class DominionTest(absltest.TestCase):
    DEFAULT_PARAMS = {"num_players": 2}

    def test_can_create_and_state(self):
        game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertIsNotNone(state)

    def test_state_rep_returns_supply_draw_discard_trash_piles_and_hands(self):
        game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertIsNotNone(state.draw_piles)
        self.assertIsNotNone(state.discard_piles)
        self.assertIsNotNone(state.hands)
        self.assertIsNotNone(state.trash_piles)
        self.assertIsNotNone(state.victory_points)

    def test_each_player_starts_with_7coppers_3_estates_in_draw_piles(self):
        game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertEqual(len(state.draw_piles), DominionTest.DEFAULT_PARAMS["num_players"])
        self.assertEqual(len(state.victory_points), DominionTest.DEFAULT_PARAMS["num_players"])
        self.assertEqual(len(state.hands), DominionTest.DEFAULT_PARAMS["num_players"])

        num_cards = lambda card_name, cards: len(list(filter(lambda card: card.name == card_name, cards)))

        for initial_draw_pile in state.draw_piles:
            self.assertEqual(num_cards('Copper', initial_draw_pile), 7)
            self.assertEqual(num_cards('Estate', initial_draw_pile), 3)

    def test_first_player_draws_5cards_from_discard_pile_to_start_game(self):
        game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertEqual(len(state.get_player(0).hand), dominion._HAND_SIZE)

    def test_isTerminal_When0ProvincesLeft(self):
        game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.victory_piles[dominion.PROVINCE.name].qty = 0
        self.assertTrue(state.is_terminal())

    def test_isTerminal_When3SupplyPilesEmpty(self):
        game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        state.kingdom_piles[dominion.VILLAGE.name].qty = 0
        state.treasure_piles[dominion.GOLD.name].qty = 0
        state.kingdom_piles[dominion.BUREAUCRAT.name].qty = 0
        self.assertTrue(state.is_terminal())


class DominionPlayerTurnTest(absltest.TestCase):
    DEFAULT_PARAMS = {"num_players": 2}

    def test_players_firstTurn_starts_0coins_0actions_1buy(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()
        observation.set_from(state, state.current_player())
        self.assertEqual(observation.dict['buys'][0], 1)
        self.assertEqual(observation.dict['actions'][0], 0)
        self.assertEqual(observation.dict['coins'][0], 0)
        self.assertEqual(observation.dict['TurnPhase'][0], dominion.TurnPhase.TREASURE_PHASE)

    def test_firstTurn_StartsTreasurePhase_InitiallegalAction_TreasureCardsAndEndPhase(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()
        actions = state.legal_actions(0)
        # initial hand will contain at least 1 copper, so expected legal actions
        # are copper and end_phase
        copper_action = 0
        observation.set_from(state, player=0)
        # player starts with 0 action cards so we skip to Treasure_Phase
        self.assertEqual(observation.dict['TurnPhase'], [dominion.TurnPhase.TREASURE_PHASE])
        self.assertEqual(actions, [copper_action, dominion.END_PHASE_ACTION]);

    def test_BuyPhase_ActionsAreThoseCardsThatCanBePurchased_AndEndPhase(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()

        # mocking BUY PHASE and coins
        player = state.get_player(0)
        player.coins = 3
        player.phase = dominion.TurnPhase.BUY_PHASE

        actions = state.legal_actions(0)
        # copper, silver, curse, estate, village, Moat, END_PHASE
        valid_actions = [0, 1, 3, 4, 7, 16, 17]
        self.assertEqual(actions, [0, 1, 3, 4, 7, 16, 17])

    def test_Action_to_String(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()

        self.assertEqual(state._action_to_string(0, 0), "Play Copper")

        player = state.get_player(0)
        player.coins = 3
        player.phase = dominion.TurnPhase.BUY_PHASE

        self.assertEqual(state._action_to_string(0, 0), "Buy and Gain Copper")

    # def test_ApplyAction_ExceptionRaised_WhenGameDone(self):
    #     game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
    #     state = game.new_initial_state()
    #     state.victory_piles[dominion.PROVINCE.name].qty = 0

    #     play_copper = 0
    #     try: 
    #         state.apply_action(play_copper)
    #     except Exception as e:
    #         self.assertEqual(str(e), "Game is finished")

    def test_ApplyAction_ExceptionRaised_WhenActionNotLegal(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        state = game.new_initial_state()

        # illegal action, player is not dealt silver initially
        play_silver = 1
        try:
            state.apply_action(play_silver)
        except Exception as e:
            self.assertEqual(str(e), "Action 1:Play Silver not in list of legal actions - 0:Play Copper, 17:End phase")

    def test_canPlay_TreasurePhase_AutoEnd(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        initial_state = game.new_initial_state()

        # play all coppers in hand
        curr_player = initial_state.get_player(initial_state.current_player())
        num_coppers = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))
        for _ in range(num_coppers):
            initial_state.apply_action(0)
        updtd_num_coppers = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))
        self.assertEqual(curr_player.coins, num_coppers)
        self.assertEqual(updtd_num_coppers, 0)
        self.assertEqual(len(curr_player.hand), dominion._HAND_SIZE - num_coppers)

        # if player plays all treasure cards, game will move onto BUY_PHASE automatically
        self.assertEqual(curr_player.phase, dominion.TurnPhase.BUY_PHASE)

    def test_canPlay_TreasurePhase_EndPhase(self):
        # #play all coppers in hand - 1 + END_PHASE
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())
        num_coppers = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))

        for _ in range(num_coppers - 1):
            state.apply_action(0)
        state.apply_action(17)
        updtd_num_coppers_in_hand = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))
        self.assertEqual(curr_player.coins, num_coppers - 1)
        self.assertEqual(updtd_num_coppers_in_hand, 1)
        self.assertEqual(curr_player.phase, dominion.TurnPhase.BUY_PHASE)

    def test_can_buy_treasure_card(self):
        # play all coppers in hand and buy a silver card
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())
        num_coppers = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))
        play_copper = 0
        buy_silver = 1
        for _ in range(num_coppers):
            state.apply_action(play_copper)

        state.apply_action(buy_silver)
        silver = dominion.SILVER.name
        self.assertIn(silver, list(map(lambda card: card.name, curr_player.draw_pile)))
        self.assertEqual(state.treasure_piles[silver].qty, 39)
        self.assertEqual(state.get_player(0).buys, 0)
        self.assertEqual(state.get_player(0).coins, num_coppers - dominion.SILVER.cost)
        self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.END_TURN)
        self.assertEqual(state.get_player(0).victory_points,3)
        self.assertEqual(state.current_player(), 1)

    def test_can_buy_victory_card(self):
        # play all coppers in hand and buy a victory card
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())
        num_coppers = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))
        play_copper = 0

        buy_estate = 4
        for _ in range(num_coppers):
            state.apply_action(play_copper)

        state.apply_action(buy_estate)
        estate = dominion.ESTATE.name

        self.assertIn(estate, list(map(lambda card: card.name, curr_player.draw_pile)))
        self.assertEqual(state.victory_piles[estate].qty, 7)
        self.assertEqual(state.get_player(0).buys, 0)
        self.assertEqual(state.get_player(0).coins, num_coppers - dominion.ESTATE.cost)
        self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.END_TURN)
        self.assertEqual(state.get_player(0).victory_points,4)
        self.assertEqual(state.current_player(), 1)

    def test_can_buy_kingdom_card(self):
        # play all coppers in hand and buy a kingdom_card
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        state = game.new_initial_state()
        curr_player = state.get_player(state.current_player())
        num_coppers = len(list(filter(lambda card: card.name is 'Copper', curr_player.hand)))
        play_copper = 0
        buy_moat = 16
        for _ in range(num_coppers):
            state.apply_action(play_copper)

        state.apply_action(buy_moat)
        moat = dominion.MOAT.name
        self.assertIn(moat, list(map(lambda card: card.name, curr_player.draw_pile)))
        self.assertEqual(state.kingdom_piles[moat].qty, 9)
        self.assertEqual(state.get_player(0).buys, 0)
        self.assertEqual(state.get_player(0).coins, num_coppers - dominion.MOAT.cost)
        self.assertEqual(state.get_player(0).phase, dominion.TurnPhase.END_TURN)
        self.assertEqual(state.get_player(0).victory_points,3)
        self.assertEqual(state.current_player(), 1)


class DominionObserverTest(absltest.TestCase):
    DEFAULT_PARAMS = {"num_players": 2}

    def test_dominion_observation(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()

        observation.set_from(state, player=0)

        self.assertEqual(list(observation.dict),
                         ["kingdom_piles", "treasure_piles", "victory_piles", "victory_points", "TurnPhase", "actions",
                          "buys",
                          "coins", "draw", "hand", "discard", "trash"])

        np.testing.assert_equal(observation.tensor.shape, (91,))

        np.testing.assert_array_equal(observation.dict["kingdom_piles"], [10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        np.testing.assert_array_equal(observation.dict["treasure_piles"], [46, 40, 30])
        np.testing.assert_array_equal(observation.dict["victory_piles"], [10, 8, 8, 8])
        np.testing.assert_array_equal(observation.dict["victory_points"], [3, 3])
        np.testing.assert_array_equal(observation.dict["coins"], [0])
        np.testing.assert_array_equal(observation.dict["buys"], [1])
        np.testing.assert_array_equal(observation.dict["actions"], [0])

        np.testing.assert_equal(len(observation.dict["draw"]), 17)
        np.testing.assert_equal(len(observation.dict["hand"]), 17)
        np.testing.assert_equal(len(observation.dict["discard"]), 17)
        np.testing.assert_equal(len(observation.dict["trash"]), 17)

        np.testing.assert_equal(np.sum(observation.dict["draw"]) + np.sum(observation.dict["hand"]), 10)
        np.testing.assert_equal(np.sum(observation.dict["discard"]), 0)
        np.testing.assert_equal(np.sum(observation.dict["trash"]), 0)

    def test_dominion_observation_str(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()

        self.assertNotEmpty(observation.string_from(state, player=0))


if __name__ == "__main__":
    absltest.main()
