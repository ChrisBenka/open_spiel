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
inline constexpr int kKingdomCards = 10;
inline constexpr int kTreasureCards = 3;
inline constexpr int kVictoryCards = 4;
inline constexpr int kInitCoppers = 7;
inline constexpr int kInitEstates = 3;
inline constexpr int kNumCards = 33;
inline constexpr int kCardsInPlay = kTreasureCards + kVictoryCards + kKingdomCards;
inline constexpr int kHandSize = 5;
inline constexpr int kGardenSupply = 8;
inline constexpr const char* kDefaultKingdomCards = "Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Workshop";
inline constexpr int kNumberStates = 5478; //todo: calculate

const enum CardType { TREASURE = 1, VICTORY = 2, ACTION = 3, ERROR = 4 };
const enum PileType { HAND = 1, DRAW = 2, DISCARD = 3, TRASH = 4, IN_PLAY = 5 };

class Effect;
class EffectRunner;
class DominionState;
class DominionObserver;


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
    Action GetReveal() const {return reveal_;}
    Action GetPlaceOntoDeck() const {return place_onto_deck_;}

    int GetCost() const {return cost_;};
    std::string ActionToString(Action action_id) const { return action_strs_.find(action_id)->second;};
    virtual CardType getCardType() const { return ERROR;};
    virtual int GetCoins() const {}; 
    virtual int GetAddActions() const {};
    virtual int GetAddBuys() const {};
    virtual int GetAddCards() const {};
    virtual int GetVictoryPoints() const {};    
    virtual const std::function<int(std::list<const Card*>)> GetVictoryPointsFn() const {};
    virtual Effect* const GetEffect() const {};
  protected:
    Action id_;
    std::string name_;
    Action play_;
    Action buy_;
    Action discard_;
    Action trash_;
    Action gain_;
    Action reveal_;
    Action place_onto_deck_;
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
    VictoryCard(int id, std::string name, int cost, int victory_points,std::function<int(std::list<const Card*>)> vp_fn) : 
    victory_points_(victory_points), vp_fn_(vp_fn), Card(id,name,cost) {};
    VictoryCard(int id, std::string name, int cost, int victory_points) : 
    victory_points_(victory_points), Card(id,name,cost) {};
    CardType getCardType() const {return VICTORY;};
    int GetVictoryPoints() const {return victory_points_;}
    const std::function<int(std::list<const Card*>)> GetVictoryPointsFn() const {return vp_fn_;}
  private:
    int victory_points_;
    std::function<int(std::list<const Card*>)> vp_fn_;
};


class ActionCard : public Card {
  public:
    ActionCard(int id, std::string name, int cost, int add_actions=0, int add_buys=0,int coins=0, int add_cards=0): 
    add_actions_(add_actions), add_buys_(add_buys),add_cards_(add_cards), coins_(coins), Card(id,name,cost), effect_ (nullptr) {};
    ActionCard(int id, std::string name, int cost, int add_actions, int add_buys,int coins, int add_cards, Effect* const effect): 
    add_actions_(add_actions), add_buys_(add_buys),add_cards_(add_cards), coins_(coins), effect_(effect), Card(id,name,cost) {};
    CardType getCardType() const {return ACTION;};
    int GetCoins() const { return coins_ ;};
    int GetAddActions() const { return add_actions_; }
    int GetAddBuys() const { return add_buys_; }
    int GetAddCards() const { return add_cards_;}
    Effect* const GetEffect() const {return effect_;};
  private:
    int add_actions_;
    int add_buys_;
    int add_cards_;
    int coins_;
    Effect* const effect_;
};

class SupplyPile {
  public: 
    SupplyPile(const Card* card, int qty) : card_(card), qty_(qty) {}; 
    bool Empty()const {return qty_ == 0;}
    int getQty(){return qty_;}
    const Card* getCard() {return card_; }
    void RemoveCardFromSupplyPile() {qty_ -= 1;}
    void Clear() {qty_ = 0;}
  private:
    int qty_;
    const Card* card_;
};

enum TurnPhase {ActionPhase, TreasurePhase, BuyPhase, EndTurn };
constexpr char * TurnPhaseStrings[] = { "Action Phase", "Treasue Phase", "Buy Phase", "End Turn Phase"};
constexpr char * SupplyPileStrings[] = { "TreasureSupply", "VictorySupply", "KingdomSuppply"};
constexpr char * PileStrings[] = { "Hand", "Draw", "Discard","Trash","InPlay"};

class PlayerState {
  public:
    PlayerState(Player id) : id_(id) {};
    int victory_points() const;
    Player GetId() const {return id_;};
    std::list<const Card*> GetAllCards() const;
    std::list<const Card*> GetDrawPile() const {return draw_pile_;};
    std::list<const Card*> GetDiscardPile() const { return discard_pile_; }
    std::list<const Card*> GetHand() const {return hand_;};
    std::list<const Card*> GetTrashPile() const {return trash_pile_;};
    std::list<const Card*> GetPile(PileType pile) const;
    bool GetAddDiscardPileToDrawPile() const { return add_discard_pile_to_draw_pile_;}; 
    void SetAddDiscardPileToDrawPile(bool add_to_draw){add_discard_pile_to_draw_pile_ = add_to_draw;};
    void SetNumRequiredCards(int num) {num_required_cards_ = num;};
    int GetNumRequiredCards() const { return num_required_cards_;}
    int GetActions() const {return actions_;}
    int GetBuys() const {return buys_;}
    int GetCoins() const {return coins_;}
    int GetVictoryPoints() const ;
    TurnPhase GetTurnPhase() const {return turn_phase_;}
    void AddToPile(const Card* card, PileType pile);
    void AddFrontToDrawPile(const Card* card){draw_pile_.push_front(card);};
    void DrawHand(int num_cards);
    const Card* TopOfDeck() const;
    bool HasCardInHand(const Card* card) const;
    bool HasTreasureCardInHand() const;
    bool HasActionCardsInHand() const;
    void PlayTreasureCard(const Card* card);
    void BuyCard(const Card* card);
    void PlayActionCard(DominionState& state, const Card* card,PileType source);
    void SetTurnPhase(TurnPhase phase){turn_phase_ = phase;}
    TurnPhase EndPhase();
    void EndTurn();
    void addCoins(int coins){coins_ += coins;};
    void RemoveFromPile(const Card* card, PileType pile);
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
  DominionState(std::shared_ptr<const Game> game, std::string kingdom_cards);
  DominionState(const DominionState&) = default;
  DominionState& operator=(const DominionState&) = default;
  Player CurrentPlayer() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  bool GameFinished() const;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  void DoApplyAction(Action move) override;
  std::map<std::string,SupplyPile> getSupplyPiles() const {return supply_piles_;}
  std::map<std::string,SupplyPile>& getSupplyPiles() {return supply_piles_;}
  void RemoveCardFromSupplyPile(std::string name) {supply_piles_.at(name).RemoveCardFromSupplyPile();}
  std::vector<PlayerState>&  getPlayers()  {return players_;}
  std::vector<PlayerState>  GetPlayers() const  {return players_;}

  PlayerState& GetCurrentPlayerState() {return players_.at(current_player_);};
  const PlayerState& GetPlayerState(Player id) const {return players_.at(id);};
  std::vector<std::string> GetKingdomCards()const {return kingdom_cards_;};
  EffectRunner* const GetEffectRunner() const {return effect_runner_;}
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
  bool EachPlayerReceivedInitSupply() const;
  bool AddDiscardPileToDrawPile() const;
  void DoApplyChanceAction(Action action_id);
  Player CurrentEffectPlayer() const;

  friend class DominionObserver;
  std::vector<std::string> kingdom_cards_;
  Player current_player_ = 0;        
  std::map<std::string,SupplyPile> supply_piles_;
  std::vector<PlayerState> players_ {PlayerState(0),PlayerState(1)};
  std::array<Effect*,kNumPlayers> effects_ = {nullptr,nullptr};
  bool is_terminal_ = false;
  EffectRunner* const effect_runner_;
};

class Effect{
  public:
    Effect(int id, std::string prompt) : id_(id), prompt_(prompt) {};
    virtual void Run(DominionState& state, PlayerState& PlayerState) {};
    virtual std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const {};
    virtual void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState) {};
    virtual std::string ActionToString(Action action, const DominionState& state, const PlayerState& PlayerState) const {return "";};
    virtual std::string ToString() const {return prompt_;}
    virtual void DoPostApplyChanceOutcomeAction(DominionState& state, PlayerState& player) {};
    int GetId() const {return id_;}
    std::string GetPrompt() const { return prompt_;}
  protected:
    int id_;
    std::string prompt_;
};

struct CardEffect{
  Effect* effect_;
};


class EffectRunner {
  public:
    EffectRunner(){
      CardEffect no_effect;
      no_effect.effect_ = nullptr;
      for(int i = 0; i < kNumPlayers; i++){
        effects_[i] = no_effect;
      }
    };
    Player CurrentPlayer() const {
      for(int i = 0; i < kNumPlayers; i++){
        if(effects_[i].effect_ != nullptr){
          return i;
        }
      }
      return 9;
    }
    Effect* GetEffect(Player player) const {
      return effects_[player].effect_;
    }
    bool Active() const {
      for(CardEffect cardEffect : effects_){
        if(cardEffect.effect_ != nullptr){
          return true;
        }
      }
      return false;
    }
    Effect* const CurrentEffect() const {
      for(CardEffect cardEffect : effects_){
        if(cardEffect.effect_ != nullptr){
          return cardEffect.effect_;
        }
      }
      return nullptr;
    }
    void AddEffect(Effect* const effect, Player player){
      CardEffect card_effect;
      card_effect.effect_ = effect;
      effects_.at(player) = card_effect;
    }
    void RemoveEffect(Player player){
      CardEffect card_effect;
      card_effect.effect_ = nullptr;
      effects_[player] = card_effect;
    }
    int Target() const {return target_;};
    void PrintEffects() const {
      for(const CardEffect effect : effects_){
        if(effect.effect_){
          std::cout << "0";
        }else{
          std::cout << "1";
        }
      }
    }
  private:
    Player target_ = 1;
    std::array<CardEffect,kNumPlayers> effects_ ;
};



class CellarEffect : public Effect {
  /* Discard any number of cards, then draw that many */
  public:
    CellarEffect(int id, std::string prompt) : Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    int num_cards_discarded_ = 0;
};

class TrashCardsEffect : public Effect {
  /* Trash up to n cards. E.g. Chapel */
  public:
    TrashCardsEffect(int id, std::string prompt, int num_cards) : num_cards_(num_cards), Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    int num_cards_trashed_ = 0;
    int num_cards_ = 0;
};

class DiscardDownToEffect : public Effect {
  /* Discard down to some number of cards in player's hand. */
  public:
    DiscardDownToEffect(int id, std::string prompt, int num_cards_down_to) : num_cards_down_to_(num_cards_down_to), Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    int num_cards_down_to_ = 0;
};

class OpponentsDiscardDownToEffect : public Effect {
    /* Causes opponents to discard down to k num cards. Opponents have option to play moat if available. E.g. Militia. */
  public:
    OpponentsDiscardDownToEffect(int id, std::string prompt, int num_cards_down_to) : num_cards_down_to_(num_cards_down_to), Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    int num_cards_down_to_ = 0;
};

class OpponentsGainCardEffect : public Effect {
    /* Causes opponents to gain a card to discard pile. Opponents have option to play moat if available. E.g. Witch. */
  public:
    OpponentsGainCardEffect(int id, std::string prompt, const Card* card) : card_(card), Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    const Card* card_;
};

class GainCardToDicardPileEffect : public Effect {
    /* Causes opponents to gain a card to discard pile. E.g. Witch. */
  public:
    GainCardToDicardPileEffect(int id, std::string prompt, const Card* card) : card_(card), Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
 private:
    const Card* card_;
};

class ChoosePileToGainEffect : public Effect {
    /* Choose a pile to gain a card from. CArd's cost must be <= n coins. E.g Workshop */
  public:
    ChoosePileToGainEffect(int id, std::string prompt, int n_coins) : n_coins_(n_coins), Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
 private:
    const int n_coins_;
};

class GainTreasureCardEffect : public Effect {
  public:
    GainTreasureCardEffect(int id, std::string prompt, const Card* card) : card_(card), Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
 private:
    const Card* card_;
};

class BanditEffect : public Effect {
    /*Gain a gold. Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper and discards the rest*/
  public:
    BanditEffect(int id, std::string prompt) : Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
};

class OpponentRevalAndTrashTopTwoCards : public Effect {
    /*Gain a gold. Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper and discards the rest*/
  public:
    OpponentRevalAndTrashTopTwoCards(int id, std::string prompt) : Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoPostApplyChanceOutcomeAction(DominionState& state, PlayerState& PlayerState);
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    int NumberTrashableCards() const;
    std::vector<const Card*> top_two_;
    bool isTrashable(const Card* c) const;
};

class TrashAndGainEffect : public Effect {
  // Trash a card fron your hand. Gain a card costing up to x coins more than it.
  public:
    TrashAndGainEffect(int id, std::string prompt,int n_coins) : n_coins_(n_coins), Effect(id,prompt) {};
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    int n_coins_;
    const Card* trashed_card_ = nullptr;
};

class TrashCardAndGainCoins : public Effect {
  // Trash a particular card from your hand for n_coins.
  public:
    TrashCardAndGainCoins(int id, std::string prompt,const Card* card, int n_coins): card_to_trash_(card), n_coins_(n_coins), Effect(id,prompt) {}; 
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    const Card* card_to_trash_;
    int n_coins_;
};

class PoacherEffect : public Effect {
  // Discard a card per empty supply pile
  public:
    PoacherEffect(int id, std::string prompt): Effect(id,prompt) {}; 
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    int num_empty_piles_;
    int num_cards_discarded_;
};

class TrashTreasureAndGainTreasure : public Effect {
  public:
    TrashTreasureAndGainTreasure(int id, std::string prompt, int n_coins): n_coins_(n_coins), Effect(id,prompt) {}; 
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
  private:
    const Card* trashed_card_ = nullptr;
    int n_coins_;
};

class VassalEffect : public Effect {
  public:
    VassalEffect(int id, std::string prompt): Effect(id,prompt) {}; 
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);
};

class DrawCardsEffect : public Effect {
  public:
    DrawCardsEffect(int id, std::string prompt, int n_cards): n_cards_(n_cards), Effect(id,prompt) {}; 
    void Run(DominionState& state, PlayerState& PlayerState);
  private:
    int n_cards_;
};

class ArtisanEffect : public Effect {
  public:
    ArtisanEffect(int id, std::string prompt, int n_coins): n_coins_(n_coins), Effect(id,prompt) {}; 
    void Run(DominionState& state, PlayerState& PlayerState);
    std::vector<Action> LegalActions(const DominionState& state, const PlayerState& PlayerState) const;
    void DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState);  
  private:
    int n_coins_;
    const Card* card_gained_ = nullptr;
};

// Game object.
class DominionGame : public Game {
 public:
  explicit DominionGame(const GameParameters& params);
  int NumDistinctActions() const override { return 0; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new DominionState(shared_from_this(),/*kingdom_cards=*/kingdom_cards_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const;
  int MaxGameLength() const override { return 2000; }

  std::shared_ptr<Observer> MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params
  ) const;
  std::string kingdom_cards_;
  std::shared_ptr<DominionObserver> default_observer_;
};


inline int GardensVpFn(std::list<const Card*> cards)  {
  return std::floor(cards.size() / 10);
}


const TreasureCard COPPER(1,"Copper",0,1);
const TreasureCard SILVER(2,"Silver",3,2);
const TreasureCard GOLD(3,"Gold",6,2);

const VictoryCard CURSE(4,"Curse",6,-1);
const VictoryCard ESTATE(5,"Estate",2,1);
const VictoryCard DUCHY(6,"Duchy",5,3);
const VictoryCard PROVINCE(7,"Province",8,6);

inline CellarEffect CELLAR_EFFECT(11,"Discard any number of cards, then draw that many.");
inline TrashCardsEffect CHAPEL_EFFECT(1,"Trash up to 4 cards",4);
inline DiscardDownToEffect MILITIA_EFFECT_DISCARD(3,"Discard down to 3 cards",3);
inline OpponentsDiscardDownToEffect MILITIA_EFFECT(2,"Opponents discard down to 3 cards",3);
inline GainCardToDicardPileEffect WITCH_GAIN_EFFECT(5,"Gain curse to discard pile",&CURSE);
inline OpponentsGainCardEffect WITCH_EFFECT(4,"Opponents gain a curse card",&CURSE);
inline ChoosePileToGainEffect WORKSHOP_EFFECT(6,"Gain a card costing up to 4 coins",4);
inline GainTreasureCardEffect GAIN_GOLD(7,"Gain a gold to discard pile",&GOLD);
inline OpponentRevalAndTrashTopTwoCards TRASH_TOP_TWO(9,"Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper, and discards the rest.");
inline BanditEffect BANDIT_EFFECT(8,"Gain a Gold. Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper, and discards the rest.");
inline TrashAndGainEffect REMODEL_EFFECT(10,"Trash a card from your hand. Gain a card costing up to 2 coins more than it.",2);
inline TrashCardAndGainCoins MONEYLENDER_EFFECT(11,"You may tash a Copper from your hand for 3 coins.",&COPPER,3);
inline PoacherEffect POACHER_EFFECT(12,"Discard a card per empty Supply pile");
inline TrashTreasureAndGainTreasure MINE_EFFECT(13,"Trash a Treasure card from your hand. Gain a Treasure card costing up to 3 coins more; put it into your hand.",3);
inline VassalEffect VASSAL_EFFECT(14,"Discard the top of your deck. If it's an Action card, you play it.");
inline DrawCardsEffect COUNCIL_ROOM_EFFECT(15,"Each other player draws a card",1);
inline ArtisanEffect ARTISAN_EFFECT(16,"Gain a card to your hand costing up to 5 coins. Put a card from your hand onto your deck.",5);


const ActionCard VILLAGE(8,"Village",3,2,0,0,1);
const ActionCard LABORATORY(9,"Laboratory",5,1,0,0,2);
const ActionCard FESTIVAL(10,"Festival",5,2,1,2,0);
const ActionCard MARKET(11,"Market",5,1,1,1,1);
const ActionCard SMITHY(12,"Smithy",4,0,0,0,3);
const ActionCard MILITIA(13,"Militia",4,0,0,0,0,&MILITIA_EFFECT);
const VictoryCard GARDENS(14,"Gardens",4,0,GardensVpFn);
const ActionCard CHAPEL(15,"Chapel",2,0,0,0,0,&CHAPEL_EFFECT);
const ActionCard WITCH(16,"Witch",5,0,0,0,2,&WITCH_EFFECT);
const ActionCard WORKSHOP(17,"Workshop",3,0,0,0,0,&WORKSHOP_EFFECT);
const ActionCard BANDIT(18,"Bandit",5,0,0,0,0,&BANDIT_EFFECT);
const ActionCard REMODEL(19,"Remodel",4,0,0,0,0,&REMODEL_EFFECT);
const ActionCard THRONE_ROOM(20,"Throne Room",4,0,0,0,0);
const ActionCard MONEYLENDER(21,"Moneylender",4,0,0,0,0,&MONEYLENDER_EFFECT);
const ActionCard POACHER(22,"Poacher",4,1,0,1,1,&POACHER_EFFECT);
const ActionCard MERCHANT(23,"Merchant",3,1,0,0,1);
const ActionCard CELLAR(24,"Cellar",2,1,0,0,0,&CELLAR_EFFECT);
const ActionCard MINE(25,"Mine",5,0,0,0,0,&MINE_EFFECT);
const ActionCard VASSAL(26,"Vassal",3,0,0,2,0,&VASSAL_EFFECT);
const ActionCard COUNCIL_ROOM(27,"Council Room",5,0,1,0,4,&COUNCIL_ROOM_EFFECT);
const ActionCard ARTISAN(28,"Artisan",6,0,0,0,0,&ARTISAN_EFFECT);
const ActionCard BUREAUCRAT(29,"Bureaucrat",4);
const ActionCard SENTRY(30,"Sentry",5,1,0,0,1);
const ActionCard HARBINGER(31,"Harbinger",3,1,0,0,1);
const ActionCard LIBRARY(32,"Library",5,0,0,0,0);
const ActionCard MOAT(33,"Moat",2,0,0,0,2);

inline constexpr Action END_PHASE_ACTION = 0;

const std::vector<const Card*> all_cards = {&COPPER,&SILVER,&GOLD,&CURSE,&ESTATE,&DUCHY,&PROVINCE,&VILLAGE,
&LABORATORY,&FESTIVAL,&MARKET,&SMITHY,&MILITIA,&GARDENS,&CHAPEL,&WITCH,&WORKSHOP,&BANDIT,
&REMODEL,&THRONE_ROOM,&MONEYLENDER,&POACHER,&MERCHANT,&CELLAR,&MINE,&VASSAL,&COUNCIL_ROOM,
&ARTISAN,&BUREAUCRAT,&SENTRY,&HARBINGER,&LIBRARY,&MOAT};

inline const Card* GetCard(Action action_id) {
  return all_cards.at((action_id - 1) % all_cards.size());
}

struct DominionObservation {
  std::vector<int> cards_in_play;
  std::vector<int> treasure_supply;
  std::vector<int> victory_supply;
  std::vector<int> kingdom_supply;
  int phase;
  int actions;
  int buys;
  int coins;
  int effect;
  std::vector<int> hand;
  std::vector<int> draw;
  std::vector<int> discard;
  std::vector<int> trash;

};

}  // namespace dominion
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DOMINION_H_
