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


from absl.testing import absltest
from open_spiel.python.games import dominion

import pyspiel

class DominionTest(absltest.TestCase):
 DEFAULT_PARAMS = {"num_players":2}
  def test_can_create_and_state(self):
    game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
    state = game.new_initial_state()
    self.assertIsNotNone(state)
  def test_state_rep_returns_supply_piles_players_deck_hand_discard_trash_pile(self):
    game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
    state = game.new_initial_state()
    self.assertIn('supply_piles',state)
    self.assertIn('deck',state)
    self.assertIn('discard',state)
    self.assertIn('hand',state)
    self.assertIn('trash',state)
  
  def test_each_player_starts_with_7coppers_3_estates_in_deck(self):
    game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
    state = game.new_initial_state()
    self.assertIn('copper',state.deck)
    self.assertIn('estate',state.deck)
    self.assertEqual(state.deck.copper.qty,7)
    self.assertEqual(state.deck.estate.qty,3)

  def test_each_player_starts_with_5_cards_in_hand(self):
    game = dominion.DominionGame(DominionTest.DEFAULT_PARAMS)
    state = game.new_initial_state()
    self.assertIn('copper',state.deck)
    self.assertIn('estate',state.deck)
    self.assertEqual(state.deck.copper.qty,7)
    self.assertEqual(state.deck.estate.qty,3)


if __name__ == "__main__":
 absltest.main()
