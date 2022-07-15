// // Copyright 2022 DeepMind Technologies Limited
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //      http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

#ifndef OPEN_SPIEL_GAMES_DOMINION_H_
#define OPEN_SPIEL_GAMES_DOMINION_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Simple game of Noughts and Crosses:
// https://en.wikipedia.org/wiki/Tic-tac-toe
//
// Parameters: none

namespace open_spiel {
namespace dominion {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kInitSupply = 10;
inline constexpr int kInitCoppers = 7;
inline constexpr int kInitEstates = 3;
inline constexpr int kHandSize = 5;

inline constexpr int kNumCards = 33;
inline constexpr int kNumRows = 3;
inline constexpr int kNumCols = 3;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCellStates = 1 + kNumPlayers;  // empty, 'x', and 'o'.

// https://math.stackexchange.com/questions/485752/Dominion-state-space-choose-calculation/485852
inline constexpr int kNumberStates = 5478;

const enum CardType {TREASURE = 1, VICTORY = 2, ACTION = 3, ERROR = 4};

class Card {
  public:
    Card(int id, std::string name, int cost);
    std::string getName() const { return name_; };
    Action GetId() const {return id_; }
    Action GetPlay() const  {return play_; }
    Action GetBuy() const {return buy_; }
    Action GetDiscard() const {return discard_;}
    Action GetTrash() const{ return trash_;}
    Action GetGain() const {return gain_;}
    int GetCost() const {return cost_;};
    std::string ActionToString(Action action_id) const { return action_strs_.find(action_id)->second;};
    virtual CardType getCardType() const { return ERROR;};
    virtual int GetCoins() const {}; 
    virtual int GetAddActions() const {};
    virtual int GetAddBuys() const {};
    virtual int GetAddCards() const {};
    virtual int GetVictoryPoints() const {};    
  protected:
    Action id_;
    std::string name_;
    Action play_;
    Action buy_;
    Action discard_;
    Action trash_;
    Action gain_;
    Action reveal_;
    int cost_;
    std::unordered_map<int,std::string> action_strs_;
};

class TreasureCard : public Card {
  public:
    TreasureCard(int id, std::string name, int cost, int coins) : coins_(coins), Card(id,name,cost) {}
    CardType getCardType() const {return TREASURE;};
    int GetCoins() const {return coins_;};
  private:
    int coins_;
};
class VictoryCard : public Card {
  public:
    VictoryCard(int id, std::string name, int cost, int victory_points) : 
    victory_points_(victory_points), Card(id,name,cost) {};
    CardType getCardType() const {return VICTORY;};
    int GetVictoryPoints() const {return victory_points_;}
  private:
    int victory_points_;
};
class ActionCard : public Card {
  public:
    ActionCard(int id, std::string name, int cost, int add_actions=0, int add_buys=0,int coins=0, int add_cards=0): 
    add_actions_(add_actions), add_buys_(add_buys),add_cards_(add_cards), coins_(coins), Card(id,name,cost) {};
    CardType getCardType() const {return ACTION;};
    int GetCoins() const { return coins_ ;};
    int GetAddActions() const { return add_actions_; }
    int GetAddBuys() const { return add_buys_; }
    int GetAddCards() const { return add_cards_;}
  private:
    int add_actions_;
    int add_buys_;
    int add_cards_;
    int coins_;
};

const TreasureCard COPPER(0,"Copper",0,1);
const TreasureCard SILVER(1,"Silver",3,2);
const TreasureCard GOLD(2,"Gold",6,2);

const VictoryCard CURSE(3,"Curse",6,-1);
const VictoryCard ESTATE(4,"Estate",2,1);
const VictoryCard DUCHY(5,"Duchy",5,3);
const VictoryCard PROVINCE(6,"Province",8,6);

const ActionCard VILLAGE(7,"Village",3,2,0,0,1);
const ActionCard LABORATORY(8,"Laboratory",5,1,0,0,2);
const ActionCard FESTIVAL(9,"Festival",5,2,1,2,0);
const ActionCard MARKET(10,"Market",5,1,1,1,1);
const ActionCard SMITHY(11,"Smithy",4,0,0,0,3);
const ActionCard MILITIA(12,"Militia",4,0,0,0,0);
const VictoryCard GARDENS(13,"Gardens",4,0);
const ActionCard CHAPEL(14,"Chapel",2,0,0,0,0);
const ActionCard WITCH(15,"Witch",5,0,0,0,2);
const ActionCard WORKSHOP(16,"Workshop",3,0,0,0,0);
const ActionCard BANDIT(17,"Bandit",5,0,0,0,0);
const ActionCard REMODEL(18,"Remodel",4,0,0,0,0);
const ActionCard THRONE_ROOM(19,"Throne Room",4,0,0,0,0);
const ActionCard MONEYLENDER(20,"Moneylender",4,0,0,0,0);
const ActionCard POACHER(21,"Poacher",4,1,0,1,1);
const ActionCard MERCHANT(22,"Merchant",3,1,0,0,1);
const ActionCard CELLAR(23,"Cellar",2,1,0,0,0);
const ActionCard MINE(24,"Mine",5,0,0,0,0);
const ActionCard VASSAL(25,"Vassal",3,0,0,2,0);
const ActionCard COUNCIL_ROOM(26,"Council Room",5,1,4,0,0);
const ActionCard ARTISAN(27,"Artisan",6,0,0,0,0);
const ActionCard BUREAUCRAT(28,"Bureaucrat",4);
const ActionCard SENTRY(29,"Sentry",5,1,0,0,1);
const ActionCard HARBINGER(30,"Harbinger",3,1,0,0,1);
const ActionCard LIBRARY(31,"Library",5,0,0,0,0);
const ActionCard MOAT(32,"Moat",2,0,0,0,2);

inline constexpr Action END_PHASE_ACTION = 167;

const std::vector<const Card*> all_cards = {&COPPER,&SILVER,&GOLD,&CURSE,&ESTATE,&DUCHY,&PROVINCE,&VILLAGE,
&LABORATORY,&FESTIVAL,&MARKET,&SMITHY,&MILITIA,&GARDENS,&CHAPEL,&WITCH,&WORKSHOP,&BANDIT,
&REMODEL,&THRONE_ROOM,&MONEYLENDER,&POACHER,&MERCHANT,&CELLAR,&MINE,&VASSAL,&COUNCIL_ROOM,
&ARTISAN,&BUREAUCRAT,&SENTRY,&HARBINGER,&LIBRARY,&MOAT};

class SupplyPile {
  public: 
    SupplyPile(const Card* card, int qty) : card_(card), qty_(qty) {}; 
    bool isEmpty()const {return qty_ == 0;}
    int getQty(){return qty_;}
    const Card* getCard() {return card_; }
    void RemoveCardFromSupplyPile() {qty_ -= 1;}
  private:
    int qty_;
    const Card* card_;
};

enum TurnPhase {ActionPhase, TreasurePhase, BuyPhase, EndTurn };
static const char * TurnPhaseStrings[] = { "Action Phase", "Treasue Phase", "Buy Phase", "End Turn Phase"};
class PlayerState {
  public:
    PlayerState(Player id) : id_(id) {};
    int victory_points() const;
    Player GetId() const {return id_;};
    std::list<const Card*> GetAllCards() const;
    std::list<const Card*> GetDrawPile() const {return draw_pile_;};
    std::list<const Card*> GetDiscardPile() const { return discard_pile_; }
    std::list<const Card*> GetHand() const {return hand_;};
    bool GetAddDiscardPileToDrawPile() const { return add_discard_pile_to_draw_pile_;}; 
    void SetAddDiscardPileToDrawPile(bool add_to_draw){add_discard_pile_to_draw_pile_ = add_to_draw;};
    int GetNumRequiredCards() const { return num_required_cards_;}
    int GetActions() const {return actions_;}
    int GetBuys() const {return buys_;}
    int GetCoins() const {return coins_;}
    int GetVictoryPoints() const ;
    TurnPhase GetTurnPhase() const {return turn_phase_;}
    void AddToHand(const Card* card) {hand_.push_back(card);}
    void AddToDrawPile(const Card* card);
    void AddFrontToDrawPile(const Card* card){draw_pile_.push_front(card);};
    void DrawHand(int num_cards);
    bool HasCardInHand(Card card) const;
    bool HasTreasureCardInHand() const;
    bool HasActionCardsInHand() const;
    void PlayTreasureCard(const Card* card);
    void BuyCard(const Card* card);
    void PlayActionCard(const Card* card);
    void SetTurnPhase(TurnPhase phase){turn_phase_ = phase;}
    TurnPhase EndPhase();
    void EndTurn();
    void addCoins(int coins){coins_ += coins;};
    void RemoveFromDiscardPile(const Card* card);
    void RemoveFromDrawPile(const Card* card);
  private:
    void AddHandInPlayCardsToDiscardPile();
    Player id_;
    std::list<const Card*> draw_pile_;
    std::list<const Card*> hand_;
    std::list<const Card*> discard_pile_;
    std::list<const Card*> trash_pile_;
    std::list<const Card*> cards_in_play_;    
    int actions_ = 1;
    int buys_ = 1;
    int coins_ = 0;
    bool add_discard_pile_to_draw_pile_ = false;
    int num_required_cards_ = 0;
    TurnPhase turn_phase_ = TreasurePhase;
};


// State of an in-play game.
class DominionState : public State {
 public:
  DominionState(std::shared_ptr<const Game> game);

  DominionState(const DominionState&) = default;
  DominionState& operator=(const DominionState&) = default;
  Player CurrentPlayer() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  bool GameFinished() const;
  std::vector<double> Returns() const override;
  // std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  void DoApplyAction(Action move) override;
  std::map<std::string,SupplyPile> getSupplyPiles()const {return supply_piles_;}
  std::vector<PlayerState>  getPlayers()  {return players_;}
  PlayerState& GetCurrentPlayerState() {return players_.at(current_player_);};
  PlayerState& GetPlayerState(Player id) {return players_.at(id);};
 private:
  std::vector<Action> LegalTreasurePhaseActions() const;
  std::vector<Action> LegalBuyPhaseActions() const;
  std::vector<Action> LegalActionPhaseActions() const;
  std::vector<std::pair<Action,double>> GetInitSupplyChanceOutcomes() const;
  std::vector<std::pair<Action, double>> GetAddDiscardPileToDrawPileChanceOutcomes() const;
  void DoApplyPlayTreasureCard(const Card* card);
  void DoApplyBuyCard(const Card* card);
  void DoApplyEndPhaseAction();
  void DoApplyInitialSupplyChanceAction(Action action_id);
  void DoApplyAddDiscardPileToDrawPile(Action action_id);
  void MoveToNextPlayer();
  Player current_player_ = 0;         // Player zero goes first
  bool EachPlayerReceivedInitSupply() const;
  bool AddDiscardPileToDrawPile() const;
  const Card* GetCard(Action action_id) const; 
  void DoApplyChanceAction(Action action_id);
  std::map<std::string,SupplyPile> supply_piles_;
  std::vector<PlayerState> players_ {PlayerState(0),PlayerState(1)};
  bool is_terminal_ = false;
};

// Game object.
class DominionGame : public Game {
 public:
  explicit DominionGame(const GameParameters& params);
  int NumDistinctActions() const override { return 0; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new DominionState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {0, 0, 0};
  }
  int MaxGameLength() const override { return kNumCells; }
};


}  // namespace dominion
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DOMINION_H_
