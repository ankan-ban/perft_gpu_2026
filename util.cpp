#include "chess.h"
#include "utils.h"
#include <string.h>

// Parse FEN string directly into QuadBitBoard + GameState + color.
// Merges the old FEN parser + board088ToQuadBB into one pass.
void readFENString(const char fen[], QuadBitBoard *qbb, GameState *gs, uint8 *color)
{
    memset(qbb, 0, sizeof(QuadBitBoard));
    gs->raw = 0;

    int i;
    int row = 0, col = 0;

    // 1. Piece placement (rank 8 down to rank 1)
    for (i = 0; fen[i]; i++)
    {
        char c = fen[i];

        if (c == '/' || c == '\\')
        {
            row++; col = 0;
        }
        else if (c >= '1' && c <= '8')
        {
            col += c - '0';
        }
        else
        {
            // Determine piece type and color from character
            uint8 piece = 0;
            uint8 pieceColor = WHITE;
            switch (c)
            {
            case 'p': piece = PAWN;   pieceColor = BLACK; break;
            case 'n': piece = KNIGHT; pieceColor = BLACK; break;
            case 'b': piece = BISHOP; pieceColor = BLACK; break;
            case 'r': piece = ROOK;   pieceColor = BLACK; break;
            case 'q': piece = QUEEN;  pieceColor = BLACK; break;
            case 'k': piece = KING;   pieceColor = BLACK; break;
            case 'P': piece = PAWN;   break;
            case 'N': piece = KNIGHT; break;
            case 'B': piece = BISHOP; break;
            case 'R': piece = ROOK;   break;
            case 'Q': piece = QUEEN;  break;
            case 'K': piece = KING;   break;
            default: break;
            }

            if (piece)
            {
                // FEN rank 8 = row 0, maps to board rank 7; rank 1 = row 7, maps to board rank 0
                int bitIndex = (7 - row) * 8 + col;
                uint64 bit = BIT(bitIndex);

                if (pieceColor == BLACK) qbb->bb[0] |= bit;
                if (piece & 1) qbb->bb[1] |= bit;
                if (piece & 2) qbb->bb[2] |= bit;
                if (piece & 4) qbb->bb[3] |= bit;

                col++;
            }
        }

        if (row >= 7 && col == 8) break;
    }

    i++;

    // 2. Active color
    while (fen[i] == ' ') i++;
    *color = (fen[i] == 'b' || fen[i] == 'B') ? BLACK : WHITE;
    i++;

    // 3. Castling availability
    while (fen[i] == ' ') i++;
    while (fen[i] != ' ')
    {
        switch (fen[i])
        {
        case 'K': gs->whiteCastle |= CASTLE_FLAG_KING_SIDE;  break;
        case 'Q': gs->whiteCastle |= CASTLE_FLAG_QUEEN_SIDE; break;
        case 'k': gs->blackCastle |= CASTLE_FLAG_KING_SIDE;  break;
        case 'q': gs->blackCastle |= CASTLE_FLAG_QUEEN_SIDE; break;
        }
        i++;
    }

    // 4. En passant
    while (fen[i] == ' ') i++;
    if (fen[i] >= 'a' && fen[i] <= 'h')
        gs->enPassent = fen[i] - 'a' + 1;

    // Skip remaining fields (halfmove clock, fullmove number)
}
