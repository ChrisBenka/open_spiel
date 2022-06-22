from cmath import phase
from json import load
from xml import dom
from absl.testing import absltest
from open_spiel.python.games import dom as dominion
import numpy as np
import pyspiel 
from collections import Counter 

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

class DominionTestStateAndGameSetup(absltest.TestCase):
    def test_can_create_game_and_state(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertIsNotNone(state)
    def test_uses_random_kingdom_cards_if_not_specifed(self):
        game = dominion.DominionGame(dominion._DEFAULT_PARAMS)
        state = game.new_initial_state()
        self.assertTrue(state.is_chance_node())
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        all_cards_in_play = set(state.supply_piles.keys())
        treasure_victory_cards = set(dominion._TREASURE_CARDS_NAMES + dominion._VICTORY_CARDS_NAMES)
        kingdom_cards = all_cards_in_play.difference(treasure_victory_cards)
        self.assertEqual(len(kingdom_cards),dominion._NUM_KINGDOM_SUPPLY_PILES)
    
    def test_throws_exception_if_invalid_kingdom_cards_provided(self):
        params = {'num_plyaers': 2, 'kingdom_cards': 'invalid'}
        try:
            game = dominion.DominionGame(params)
        except Exception as e:
            msg = str(e)
            self.assertEqual(msg,"Expected list of 10 unique kingdom cards separated by a comma")

        try:
            params['kingdom_cards'] = "invalid1, invalid2, invalid3, invalid4, invalid5, invalid6, invalid7, invalid8, invalid9, invalid10"
            game = dominion.DominionGame(params)
        except Exception as e:
            msg = 'is not an available kingdom card. Available kingdom cards: \n Village, Laboratory, Festival, Market, Smithy, Militia, Gardens, Chapel, Witch, Workshop, Bandit, Remodel, Throne Room, Moneylender, Poacher, Merchant, Cellar, Mine, Vassal, Council Room, Artisan, Bureaucrat, Sentry, Harbinger, Library, Moat'
            self.assertIn(msg,str(e))

    def test_can_create_game_using_specified_kingdom_cards(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        self.assertIn('Village',state.supply_piles)
        self.assertIn('Workshop',state.supply_piles)

    def test_each_player_starts_with_7copper_3estates(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        self.assertEqual(len(state._players[0].all_cards),dominion._INITIAL_SUPPLY)
        self.assertEqual(len(state._players[1].all_cards),dominion._INITIAL_SUPPLY)
        self.assertEqual(state.current_player(),0)

    def test_each_player_starts_with_0coins_1buy_1action(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        self.assertEqual(current_player_state(state).coins,0)
        self.assertEqual(current_player_state(state).buys,1)
        self.assertEqual(current_player_state(state).actions,1)


class DominionTestPlayTurn(absltest.TestCase):
    params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
    
    def test_treasure_phase_legal_actions(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop'}
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        self.assertEqual(state._legal_actions(0),[dominion.COPPER.play,dominion.END_PHASE_ACTION])

    def test_play_treasure_card_and_end_turn(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        state.apply_action(dominion.COPPER.play)
        self.assertIn(dominion.COPPER,current_player_state(state).cards_in_play)
        self.assertEqual(current_player_state(state).coins,1)
        state.apply_action(dominion.END_PHASE_ACTION)
        self.assertEqual(state.current_player(),0)
        self.assertEqual(current_player_state(state).phase,dominion.TurnPhase.BUY_PHASE)

    def test_play_all_treasure_cards(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        state.apply_action(dominion.END_PHASE_ACTION)
        self.assertEqual(state.current_player(),0)
        self.assertEqual(current_player_state(state).phase,dominion.TurnPhase.BUY_PHASE)
    
    def test_skip_buy_card(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.END_PHASE_ACTION)
        #moves to next player 
        self.assertEqual(state.current_player(),1)
    

    def test_buy_treasure_card(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.COPPER.buy)
        state.apply_action(dominion.END_PHASE_ACTION)
        #moves to next player 
        self.assertEqual(state.current_player(),1)
    
    def test_buy_action_card(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.VILLAGE.buy)
        self.assertIn(dominion.VILLAGE,player_state(state,0).draw_pile)
        state.apply_action(dominion.END_PHASE_ACTION)
        self.assertEqual(state.current_player(),1)

    def test_buy_victory_card_increases_vp(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.DUCHY.buy)
        self.assertIn(dominion.DUCHY,player_state(state,0).draw_pile)
        self.assertEqual(player_state(state,0).victory_points,8)
        state.apply_action(dominion.END_PHASE_ACTION)
        self.assertEqual(state.current_player(),1)

    def test_end_turn(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.DUCHY.buy)
        state.apply_action(dominion.END_PHASE_ACTION)
        self.assertEqual(player_state(state,0).actions,1)
        self.assertEqual(player_state(state,0).buys,1)
        self.assertEqual(player_state(state,0).coins,0)
    
    def test_play_treasure_card_increases_coins(self):
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        state.apply_action(dominion.COPPER.play)
        self.assertEqual(player_state(state,0).coins,1)
    

class DominionTestActionCards(absltest.TestCase):
    def test_village(self):
        """Playing village adds 1 card, 2 actions """
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.VILLAGE.buy)
        state.apply_action(dominion.END_PHASE_ACTION)
        
        self.assertEqual(state.current_player(),1)
        state.apply_action(dominion.END_PHASE_ACTION)
        load_hand(state,['Village','Copper','Copper','Estate','Estate'])

        state.apply_action(dominion.VILLAGE.play)
        self.assertEqual(len(current_player_state(state).hand),5)
        self.assertEqual(current_player_state(state).actions,2)

    def test_laboratory(self):
        """add 2 cards ; add 1 action"""
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.LABORATORY.buy)
        state.apply_action(dominion.END_PHASE_ACTION)
        
        self.assertEqual(state.current_player(),1)
        state.apply_action(dominion.END_PHASE_ACTION)

        load_hand(state,['Laboratory','Copper','Copper','Estate','Estate'])

        state.apply_action(dominion.LABORATORY.play)
        # draw pile will not contain the required minimum of 2 cards
        self.assertEqual(len(current_player_state(state).draw_pile),1)
        self.assertTrue(state.is_chance_node())
        #the discard pile will be added to the draw pile in chance order
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p = prob_list)
            state.apply_action(action)
        self.assertEqual(len(current_player_state(state).hand),6)
        self.assertEqual(current_player_state(state).actions,1)
        self.assertEqual(len(current_player_state(state).discard_pile),0)

    def test_festival(self):
        """add 2 actions; 1 buys; 2 coins ; """
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.FESTIVAL.buy)
        state.apply_action(dominion.END_PHASE_ACTION)
        
        self.assertEqual(state.current_player(),1)
        state.apply_action(dominion.END_PHASE_ACTION)

        load_hand(state,['Festival','Copper','Copper','Estate','Estate'])

        state.apply_action(dominion.FESTIVAL.play)
        self.assertEqual(current_player_state(state).actions,2)
        self.assertEqual(current_player_state(state).buys,2)
        self.assertEqual(current_player_state(state).coins,2)
    
    def test_market(self):
        """add 1 actions; 1 buy; 1 coin ; 1 card """
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.MARKET.buy)
        state.apply_action(dominion.END_PHASE_ACTION)
        
        self.assertEqual(state.current_player(),1)
        state.apply_action(dominion.END_PHASE_ACTION)

        load_hand(state,['Market','Copper','Copper','Estate','Estate'])

        state.apply_action(dominion.MARKET.play)
        self.assertEqual(current_player_state(state).actions,1)
        self.assertEqual(current_player_state(state).buys,2)
        self.assertEqual(current_player_state(state).coins,1)
    
    def test_smithy(self):
        """add 3 cards """
        params = {'num_players': 2, 'kingdom_cards': 'Village,Laboratory,Festival,Market,Smithy,Militia,Gardens,Chapel,Witch,Workshop' }
        game = dominion.DominionGame(params)
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            if prob_list[0] != 0:
                state.apply_action(dominion.COPPER.id)
            else:
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
        #draw 5 
        while current_player_state(state).has_treasure_cards_in_hand and current_player_state(state).phase is dominion.TurnPhase.TREASURE_PHASE:
            state.apply_action(dominion.COPPER.play)
        #5 coppers added to discard pile, 5 estates in draw pile. 
        state.apply_action(dominion.END_PHASE_ACTION)
        state.apply_action(dominion.SMITHY.buy)
        # SMITHY added to draw pile. -> 6 size of draw pile. 
        state.apply_action(dominion.END_PHASE_ACTION)
        
        self.assertEqual(state.current_player(),1)
        state.apply_action(dominion.END_PHASE_ACTION)
        
        load_hand(state,['Smithy','Copper','Copper','Estate','Estate'])
        # draw pile has 1 estate, discard_pile has x 
        state.apply_action(dominion.SMITHY.play)
        self.assertEqual(len(current_player_state(state).draw_pile),1)
        self.assertTrue(state.is_chance_node())
        #the discard pile will be added to the draw pile in chance order
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p = prob_list)
            state.apply_action(action)
        self.assertEqual(len(current_player_state(state).hand),7)
        self.assertEqual(len(current_player_state(state).discard_pile),0)

    
if __name__ == "__main__":
    absltest.main()
