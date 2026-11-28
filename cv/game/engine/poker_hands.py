from collections import Counter
from collections.abc import Sequence

try:
    from game.balatro_models import (
        Enhancement,
        Joker,
        JokerName,
        PlayingCard,
        PokerHand,
        Rank,
        Suit,
    )
except ImportError:
    from balatro_models import (
        Enhancement,
        Joker,
        JokerName,
        PlayingCard,
        PokerHand,
        Rank,
        Suit,
    )


RANK_VALUES: dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}


def evaluate_poker_hand(
    cards: Sequence[PlayingCard],
    jokers: Sequence[Joker] | None = None,
) -> PokerHand:
    """Evaluates the poker hand type for a set of played or selected cards.

    Fully accounts for Balatro rules including Stone cards, Wild cards,
    and modifier jokers (Four Fingers, Shortcut, Smeared Joker).
    """
    if not cards:
        return PokerHand.HIGH_CARD

    joker_names = {
        j.name for j in (jokers or []) if not getattr(j, "debuffed", False)
    }
    has_four_fingers = JokerName.FOUR_FINGERS in joker_names
    has_shortcut = JokerName.SHORTCUT in joker_names
    has_smeared = JokerName.SMEARED_JOKER in joker_names

    # Filter out Stone cards (they have no rank and no suit)
    valid_cards = [
        c for c in cards if c.enhancement != Enhancement.STONE and c.rank != Rank.NONE
    ]

    if not valid_cards:
        return PokerHand.HIGH_CARD

    rank_counts = Counter(c.rank for c in valid_cards)
    counts_sorted = sorted(rank_counts.values(), reverse=True)

    # Helper: checks if a subset of cards forms a flush
    def is_flush(card_subset: Sequence[PlayingCard], min_len: int = 5) -> bool:
        if len(card_subset) < min_len:
            return False

        # All base suits
        suits_to_check = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        for target_suit in suits_to_check:
            match_count = 0
            for c in card_subset:
                if c.enhancement == Enhancement.WILD:
                    match_count += 1
                elif has_smeared:
                    if (
                        target_suit in (Suit.HEARTS, Suit.DIAMONDS)
                        and c.suit in (Suit.HEARTS, Suit.DIAMONDS)
                        or target_suit in (Suit.SPADES, Suit.CLUBS)
                        and c.suit in (Suit.SPADES, Suit.CLUBS)
                    ):
                        match_count += 1
                elif c.suit == target_suit:
                    match_count += 1
            if match_count >= min_len:
                return True
        return False

    # Helper: checks if a set of card ranks can form a straight
    def is_straight(card_subset: Sequence[PlayingCard], min_len: int = 5) -> bool:
        if len(card_subset) < min_len:
            return False

        unique_vals = {
            RANK_VALUES[c.rank] for c in card_subset if c.rank in RANK_VALUES
        }
        if not unique_vals:
            return False

        # Ace can also act as 1 (low straight)
        if 14 in unique_vals:
            unique_vals.add(1)

        sorted_vals = sorted(unique_vals)
        max_consecutive = 1
        current_streak = 1

        for i in range(1, len(sorted_vals)):
            diff = sorted_vals[i] - sorted_vals[i - 1]
            if diff == 1 or (has_shortcut and diff == 2):
                current_streak += 1
                max_consecutive = max(max_consecutive, current_streak)
            elif diff > 0:
                current_streak = 1

        return max_consecutive >= min_len

    flush_req = 4 if has_four_fingers else 5
    straight_req = 4 if has_four_fingers else 5

    has_fl = is_flush(valid_cards, min_len=flush_req)
    has_st = is_straight(valid_cards, min_len=straight_req)

    # 1. Flush Five: 5 cards of same rank and all same suit
    if counts_sorted[0] >= 5 and is_flush(valid_cards, min_len=5):
        return PokerHand.FLUSH_FIVE

    # 2. Flush House: Full house (3 of a kind + pair) where all cards share suit
    if (
        len(valid_cards) >= 5
        and counts_sorted[0] >= 3
        and len(counts_sorted) > 1
        and counts_sorted[1] >= 2
        and is_flush(valid_cards, min_len=5)
    ):
        return PokerHand.FLUSH_HOUSE

    # 3. Five of a Kind
    if counts_sorted[0] >= 5:
        return PokerHand.FIVE_OF_A_KIND

    # 4. Straight Flush
    if has_fl and has_st:
        # Check if the flush cards specifically form the straight
        # For simplicity, if both are present in the played cards, it ranks as Straight Flush
        return PokerHand.STRAIGHT_FLUSH

    # 5. Four of a Kind
    if counts_sorted[0] >= 4:
        return PokerHand.FOUR_OF_A_KIND

    # 6. Full House (3 of one rank and 2 of another)
    if counts_sorted[0] >= 3 and len(counts_sorted) > 1 and counts_sorted[1] >= 2:
        return PokerHand.FULL_HOUSE

    # 7. Flush
    if has_fl:
        return PokerHand.FLUSH

    # 8. Straight
    if has_st:
        return PokerHand.STRAIGHT

    # 9. Three of a Kind
    if counts_sorted[0] >= 3:
        return PokerHand.THREE_OF_A_KIND

    # 10. Two Pair
    if len(counts_sorted) > 1 and counts_sorted[0] >= 2 and counts_sorted[1] >= 2:
        return PokerHand.TWO_PAIR

    # 11. Pair
    if counts_sorted[0] >= 2:
        return PokerHand.PAIR

    # 12. High Card
    return PokerHand.HIGH_CARD


def get_scoring_cards(
    cards: Sequence[PlayingCard],
    jokers: Sequence[Joker] | None = None,
) -> list[PlayingCard]:
    """Extracts the exact subset of played cards that actively score for the hand.

    Adheres strictly to Balatro game logic:
    1. If the player has Splash, all played cards score.
    2. Any Stone Card played is always included in scoring cards.
    3. For rank/suit hands, only the matching combination cards score;
       kickers/unrelated cards are excluded.
    """
    if not cards:
        return []

    joker_names = {
        j.name for j in (jokers or []) if not getattr(j, "debuffed", False)
    }
    if JokerName.SPLASH in joker_names:
        return list(cards)

    has_four_fingers = JokerName.FOUR_FINGERS in joker_names
    has_shortcut = JokerName.SHORTCUT in joker_names
    has_smeared = JokerName.SMEARED_JOKER in joker_names

    stone_cards = [
        c for c in cards if c.enhancement == Enhancement.STONE or c.rank == Rank.NONE
    ]
    valid_cards = [
        c for c in cards if c.enhancement != Enhancement.STONE and c.rank != Rank.NONE
    ]

    if not valid_cards:
        return stone_cards

    hand_type = evaluate_poker_hand(cards, jokers)
    rank_counts = Counter(c.rank for c in valid_cards)
    sorted_ranks = sorted(
        rank_counts.keys(),
        key=lambda r: (rank_counts[r], RANK_VALUES.get(r, 0)),
        reverse=True,
    )

    def get_cards_of_rank(r: Rank) -> list[PlayingCard]:
        return [c for c in valid_cards if c.rank == r]

    def find_flush_cards(min_len: int = 5) -> list[PlayingCard]:
        suits_to_check = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        for target_suit in suits_to_check:
            matched: list[PlayingCard] = []
            for c in valid_cards:
                if c.enhancement == Enhancement.WILD:
                    matched.append(c)
                elif has_smeared:
                    if (
                        target_suit in (Suit.HEARTS, Suit.DIAMONDS)
                        and c.suit
                        in (
                            Suit.HEARTS,
                            Suit.DIAMONDS,
                        )
                        or target_suit in (Suit.SPADES, Suit.CLUBS)
                        and c.suit
                        in (
                            Suit.SPADES,
                            Suit.CLUBS,
                        )
                    ):
                        matched.append(c)
                elif c.suit == target_suit:
                    matched.append(c)
            if len(matched) >= min_len:
                return matched[:5]
        return []

    def find_straight_cards(min_len: int = 5) -> list[PlayingCard]:
        val_map: dict[int, list[PlayingCard]] = {}
        for c in valid_cards:
            val = RANK_VALUES.get(c.rank, 0)
            if val > 0:
                val_map.setdefault(val, []).append(c)
        if 14 in val_map:
            val_map[1] = val_map[14]

        sorted_vals = sorted(val_map.keys(), reverse=True)
        best_seq: list[int] = []
        curr_seq: list[int] = []

        for v in sorted_vals:
            if not curr_seq:
                curr_seq = [v]
            else:
                diff = curr_seq[-1] - v
                if diff == 1 or (has_shortcut and diff == 2):
                    curr_seq.append(v)
                elif diff > 0:
                    if len(curr_seq) > len(best_seq):
                        best_seq = list(curr_seq)
                    curr_seq = [v]
        if len(curr_seq) > len(best_seq):
            best_seq = list(curr_seq)

        if len(best_seq) >= min_len:
            seq_cards: list[PlayingCard] = []
            for v in best_seq[:5]:
                card = val_map[v][0]
                if card not in seq_cards:
                    seq_cards.append(card)
            return seq_cards
        return []

    scoring: list[PlayingCard] = []

    match hand_type:
        case PokerHand.FLUSH_FIVE:
            target_rank = sorted_ranks[0]
            scoring = get_cards_of_rank(target_rank)[:5]

        case PokerHand.FLUSH_HOUSE:
            r3, r2 = sorted_ranks[0], sorted_ranks[1]
            scoring = get_cards_of_rank(r3)[:3] + get_cards_of_rank(r2)[:2]

        case PokerHand.FIVE_OF_A_KIND:
            target_rank = sorted_ranks[0]
            scoring = get_cards_of_rank(target_rank)[:5]

        case PokerHand.STRAIGHT_FLUSH:
            scoring = find_flush_cards(min_len=4 if has_four_fingers else 5)
            if not scoring:
                scoring = find_straight_cards(min_len=4 if has_four_fingers else 5)

        case PokerHand.FOUR_OF_A_KIND:
            target_rank = sorted_ranks[0]
            scoring = get_cards_of_rank(target_rank)[:4]

        case PokerHand.FULL_HOUSE:
            r3, r2 = sorted_ranks[0], sorted_ranks[1]
            scoring = get_cards_of_rank(r3)[:3] + get_cards_of_rank(r2)[:2]

        case PokerHand.FLUSH:
            scoring = find_flush_cards(min_len=4 if has_four_fingers else 5)

        case PokerHand.STRAIGHT:
            scoring = find_straight_cards(min_len=4 if has_four_fingers else 5)

        case PokerHand.THREE_OF_A_KIND:
            target_rank = sorted_ranks[0]
            scoring = get_cards_of_rank(target_rank)[:3]

        case PokerHand.TWO_PAIR:
            r1, r2 = sorted_ranks[0], sorted_ranks[1]
            scoring = get_cards_of_rank(r1)[:2] + get_cards_of_rank(r2)[:2]

        case PokerHand.PAIR:
            target_rank = sorted_ranks[0]
            scoring = get_cards_of_rank(target_rank)[:2]

        case PokerHand.HIGH_CARD | _:
            highest_card = max(valid_cards, key=lambda c: RANK_VALUES.get(c.rank, 0))
            scoring = [highest_card]

    for sc in stone_cards:
        if sc not in scoring:
            scoring.append(sc)

    return scoring
