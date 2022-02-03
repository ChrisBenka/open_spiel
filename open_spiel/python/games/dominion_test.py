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
        self.assertEqual(len(state.get_player(0).hand),dominion._HAND_SIZE)


class DominionObserverTest(absltest.TestCase):
    DEFAULT_PARAMS = {"num_players": 2}

    def test_dominion_observation(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()

        observation.set_from(state, player=0)

        self.assertEqual(list(observation.dict),
                         ["kingdom_piles", "treasure_piles", "victory_piles", "victory_points", "actions", "buys",
                          "coins","draw","hand","discard","trash"])

        np.testing.assert_equal(observation.tensor.shape,(90,))

        np.testing.assert_array_equal(observation.dict["kingdom_piles"], [10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        np.testing.assert_array_equal(observation.dict["treasure_piles"], [46, 40, 30])
        np.testing.assert_array_equal(observation.dict["victory_piles"], [10, 8, 8, 8])
        np.testing.assert_array_equal(observation.dict["victory_points"], [0, 0])
        np.testing.assert_array_equal(observation.dict["coins"], [0])
        np.testing.assert_array_equal(observation.dict["buys"], [1])
        np.testing.assert_array_equal(observation.dict["actions"], [1])

        np.testing.assert_equal(len(observation.dict["draw"]),17)
        np.testing.assert_equal(len(observation.dict["hand"]),17)
        np.testing.assert_equal(len(observation.dict["discard"]),17)
        np.testing.assert_equal(len(observation.dict["trash"]),17)

        np.testing.assert_equal(np.sum(observation.dict["draw"])+np.sum(observation.dict["hand"]),10)
        np.testing.assert_equal(np.sum(observation.dict["discard"]),0)
        np.testing.assert_equal(np.sum(observation.dict["trash"]),0)
        
    def test_dominion_observation_str(self):
        game = dominion.DominionGame(DominionObserverTest.DEFAULT_PARAMS)
        observation = game.make_py_observer()
        state = game.new_initial_state()

        self.assertNotEmpty(observation.string_from(state, player=0))



if __name__ == "__main__":
    absltest.main()
