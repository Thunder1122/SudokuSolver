# All group members were present and contributing during all work on this project! :)
# Tushar Chandra: tac311
# Trent Cwiok: tbc808
# Vyas Alwar: vaa143

#!/usr/bin/env python
import struct, string, math, pdb
import copy

CHECKS = 0
HAS_PRINTED = False

class SudokuBoard:
    """This will be the sudoku board game object your player will manipulate."""
  
    def __init__(self, size, board):
      """the constructor for the SudokuBoard"""
      self.BoardSize = size #the size of the board
      self.CurrentGameBoard= board #the current state of the game board

      self.domains = {(i, j): range(1, size + 1) for i in range(size) for j in range(size)}
      self._initialize_domains()

    def _initialize_domains(self):
        ''' Sets up the initial domains for the board '''

        BoardArray = self.CurrentGameBoard
        size = self.BoardSize
        subsquare = int(math.sqrt(size))

        # Iterate through entire board, and for each cell, we find the domains
        # for each.

        for row in range(size):            
            for col in range(size):
                if BoardArray[row][col] != 0:
                    self.change_domains(row, col)

        return

    def change_domains(self, row, col):
        ''' Given a cell at (row, col), modify the domains of all cells it 
        affects in order to eliminate the possibilities '''

        val = self.CurrentGameBoard[row][col]
        subsquare = int(math.sqrt(self.BoardSize))

        # Change domain of the cell itself to only be itself
        self.domains[(row, col)] = [val]

        # Modify the domains of all the cells in the same row and column 
        for i in range(self.BoardSize):
            # cells in same row, except itself
            if val in self.domains[(row, i)] and i != col:
                self.domains[(row, i)].remove(val)

            # cells in the same column, except itself
            if val in self.domains[(i, col)] and i != row:
                self.domains[(i, col)].remove(val)

        # Modify the domain of the cells in the same subsquare
        # square_row/col is the subsquare index
        # row/col_start is the starting row/col index of the top-left
        # square in that subsquare

        square_row = row // subsquare
        square_col = col // subsquare

        row_start = square_row * subsquare
        col_start = square_col * subsquare

        for i in range(subsquare):
            for j in range(subsquare):
                # check each cell in the subsquare, except itself
                if val in self.domains[(row_start + i, col_start + j)] and (row_start + i) != row and (col_start + j) != col:
                    self.domains[(row_start + i, col_start + j)].remove(val)

        return

    def set_value(self, row, col, value):
        """This function will create a new sudoku board object with the input
        value placed on the GameBoard row and col are both zero-indexed"""

        #add the value to the appropriate position on the board
        self.CurrentGameBoard[row][col] = value
        #return a new board of the same size with the value added
        return SudokuBoard(self.BoardSize, self.CurrentGameBoard)
                                                                  
    def print_board(self):
        """Prints the current game board. Leaves unassigned spots blank."""
        div = int(math.sqrt(self.BoardSize))
        dash = ""
        space = ""
        line = "+"
        sep = "|"
        for i in range(div):
            dash += "----"
            space += "    "
        for i in range(div):
            line += dash + "+"
            sep += space + "|"
        for i in range(-1, self.BoardSize):
            if i != -1:
                print "|",
                for j in range(self.BoardSize):
                    if self.CurrentGameBoard[i][j] > 9:
                        print self.CurrentGameBoard[i][j],
                    elif self.CurrentGameBoard[i][j] > 0:
                        print "", self.CurrentGameBoard[i][j],
                    else:
                        print "  ",
                    if (j+1 != self.BoardSize):
                        if ((j+1)//div != j/div):
                            print "|",
                        else:
                            print "",
                    else:
                        print "|"
            if ((i+1)//div != i/div):
                print line
            else:
                print sep

    def is_legal(self):
        ''' Tests to see if it is a legal board by checking 
        to see if any of the domains are empty. '''

        # Initialize. For a 9x9, size = 9, and subsquare = 3
        BoardArray = self.CurrentGameBoard
        size = self.BoardSize
        subsquare = int(math.sqrt(size))

        '''for row in range(size):
            for col in range(size):
                # if the domain of any cell is empty, it's not 
                # legal, so return False

                if BoardArray[row][col] not in self.domains[(row, col)] and BoardArray[row][col] != 0:
                    return False

        '''# Method to check duplicates: create an empty dict / hash table
        # that will contain all of the seen / present values. If a value
        # is not in the dict, add it; if it is, then we've already seen 
        # it, so it's a duplicate, so return False

        # Checks all of the rows for duplicates
        for row in range(size):
            present = {}
            
            for col in range(size):
                if BoardArray[row][col] in present:
                    return False

                elif BoardArray[row][col] != 0:
                    present[BoardArray[row][col]] = 1

        # Checks all of the columns for duplicates
        for col in range(size):
            present = {}

            for row in range(size):
                if BoardArray[row][col] in present:
                    return False

                elif BoardArray[row][col] != 0:
                    present[BoardArray[row][col]] = 1

        # Checks all of the subsquares for duplicates
        # In general, to check subsquare (i, j); i in [0, subsquare]; j in [0, subsquare];
        # with (0, 0) being top left, we need to check all of the squares from 
        # row: i*subsquare -> i*subsquare + subsquare; 
        # col: j*subsquare -> j*subsquare + subsquare.

        # rowsq = first subsquare index, colsq = second subsquare index
        for rowsq in range(subsquare):
            for colsq in range(subsquare):
                present = {}

                # i/jstart = actual row/col indices of the square on the board to start
                rowstart = rowsq * subsquare
                colstart = colsq * subsquare

                # we need rows from rowstart to rowstart + subsquare, cols likewise
                for row in range(rowstart, rowstart + subsquare):
                    for col in range(colstart, colstart + subsquare):
                        if BoardArray[row][col] in present:
                            return False

                        elif BoardArray[row][col] != 0:
                            present[BoardArray[row][col]] = 1

        return True


def parse_file(filename):
    """Parses a sudoku text file into a BoardSize, and a 2d array which holds
    the value of each cell. Array elements holding a 0 are considered to be
    empty."""

    f = open(filename, 'r')
    BoardSize = int( f.readline())
    NumVals = int(f.readline())

    #initialize a blank board
    board= [ [ 0 for i in range(BoardSize) ] for j in range(BoardSize) ]

    #populate the board with initial values
    for i in range(NumVals):
        line = f.readline()
        chars = line.split()
        row = int(chars[0])
        col = int(chars[1])
        val = int(chars[2])
        board[row-1][col-1]=val
    
    return board



def init_board(file_name):
    """Creates a SudokuBoard object initialized with values from a text file"""
    global CHECKS, HAS_PRINTED
    CHECKS = 0
    HAS_PRINTED = False
    
    board = parse_file(file_name)
    return SudokuBoard(len(board), board)

    
def is_complete(sudoku_board):
    """Takes in a sudoku board and tests to see if it has been filled in
    correctly."""
    BoardArray = sudoku_board.CurrentGameBoard
    size = len(BoardArray)
    subsquare = int(math.sqrt(size))

    #check each cell on the board for a 0, or if the value of the cell
    #is present elsewhere within the same row, column, or square
    for row in range(size):
        for col in range(size):
            if BoardArray[row][col]==0:
                return False
            for i in range(size):
                if ((BoardArray[row][i] == BoardArray[row][col]) and i != col):
                    return False
                if ((BoardArray[i][col] == BoardArray[row][col]) and i != row):
                    return False
            #determine which square the cell is in
            SquareRow = row // subsquare
            SquareCol = col // subsquare
            for i in range(subsquare):
                for j in range(subsquare):
                    if((BoardArray[SquareRow*subsquare+i][SquareCol*subsquare+j]
                            == BoardArray[row][col])
                        and (SquareRow*subsquare + i != row)
                        and (SquareCol*subsquare + j != col)):
                            return False
    return True


def get_degree(initial_board, row, col):
    '''Given a single cell in the Sudoku board, will calculate the degree for
    that cell.  The degree is defined as the number of unassigned values in
    that cells row, column, and subsquare.'''

    BoardArray = initial_board.CurrentGameBoard
    size = initial_board.BoardSize
    subsquare = int(math.sqrt(size))

    degree = 0

    for i in range(size):
        if BoardArray[i][col] == 0:
            degree += 1
        if BoardArray[row][i] == 0:
            degree += 1

    SquareRow = row // subsquare
    SquareCol = col // subsquare
    for i in range(subsquare):
        for j in range(subsquare):
            if((BoardArray[SquareRow*subsquare+i][SquareCol*subsquare+j]
                    == 0)
                and (SquareRow*subsquare + i != row)
                and (SquareCol*subsquare + j != col)):
                    degree += 1

    return degree


def choose_square(initial_board, MRV = False, Degree = False):
    ''' Chooses the next square (variable) to assign a value to, given some
    combination of the heuristics MRV (most remaining values) and Degree. 
    Takes a SudokuBoard. '''

    BoardArray = initial_board.CurrentGameBoard
    size = initial_board.BoardSize
    subsquare = int(math.sqrt(size))

    if not (MRV or Degree):
        # With no heuristics, find first zero square

        for row in range(size):
            for col in range(size):
                    if BoardArray[row][col] == 0:
                        return row, col

    if MRV:
        # With MRV, find the key associated with the fewest values
        # by iterating over domains.values(). The desired key will be
        # the one that has the shortest length domain.

        min_vals = 10

        for k in initial_board.domains.keys():
            l = len(initial_board.domains[k])
            if l < min_vals and initial_board.CurrentGameBoard[k[0]][k[1]] == 0:
                min_key = k
                min_vals = l
        
        return min_key

    if Degree and not MRV:
        # For each square, count the unassigned values in its row + col + subsquare
        # and find the one with the largest count. Only do this if we don't
        # do MRV, because MRV is better.

        max_degree = -1
        max_key = (-1, -1)

        for row in range(size):
            for col in range(size):
                if BoardArray[row][col] == 0:
                    degree = get_degree(initial_board, row, col)
                    if degree > max_degree:
                        max_degree = degree
                        max_key = (row, col)
        #initial_board.print_board()

        #print max_key, max_degree, initial_board.domains[max_key]
        return max_key


def order_domain(initial_board, row, col, domain, domains_copy, LCV):
    ''' Orders a domain for (row, col) from most desirable to least desirable, 
    with or without the LCV (least constraining value) heuristic. 
    Takes a SudokuBoard. '''

    # Without heuristics, just return the domain because we choose any value
    if not LCV:
        return domain

    if LCV:
        BoardArray = initial_board.CurrentGameBoard
        size = initial_board.BoardSize
        subsquare = int(math.sqrt(size))

        SquareRow = row // subsquare
        SquareCol = col // subsquare
        #Make a new domain to save the reordered values into
        LCV_domain = []
        #Make a dictionary that stores the value from the domain and the number of contraints it causes
        LCV_values = {}

        #Loop over all values in the domain of the chosen cell
        for check_val in domain:
            constraints = 0
            for i in range(size):
                #Only consider unassigned cells, as assigned cells cannot be "constrained"
                if BoardArray[i][col] == 0:
                    #Loop over the row and col of the chosen cell and count how many other domains
                    #the value being looked at currently would constrain
                    if check_val in domains_copy[(i, col)] and i != row:
                        constraints += 1

                if BoardArray[row][i] == 0:
                    if check_val in domains_copy[(row, i)] and i != col:
                        constraints += 1

            for i in range(subsquare):
                for j in range(subsquare):
                    #Again, only consider unassigned cells
                    if BoardArray[SquareRow * subsquare + i][SquareCol * subsquare + j] == 0:
                        #Loop over the subsquare and count constraints in the same way
                        if check_val in domains_copy[(SquareRow*subsquare+i, SquareCol*subsquare+j)] and (SquareRow*subsquare + i != row and (SquareCol*subsquare + j != col)):
                                    constraints += 1

            #Add the value being looked at to the dictionary as a key, with the number of constraints being its value
            LCV_values[check_val] = constraints

        #print LCV_values
        #Append the key (a (row, col) tuple, or cell) with the lowest value to the domain, then delete it from the dict
        #This will order the domain with the least constraining value first, and the most constraining value last
        while len(LCV_values) > 0:
            LCV_domain.append(min(LCV_values, key=LCV_values.get))
            del LCV_values[min(LCV_values, key=LCV_values.get)]

        #Return the newly ordered domain
        return LCV_domain


def solve(initial_board, forward_checking = False, MRV = False, Degree = False,
    LCV = False):
    """Takes an initial SudokuBoard and solves it using back tracking, and zero
    or more of the heuristics and constraint propagation methods (determined by
    arguments). Returns the resulting board solution. """

    size = initial_board.BoardSize

    # If it's complete, we're done
    if is_complete(initial_board):
        return initial_board

    # Choose a square based on some heuristics
    row, col = choose_square(initial_board, MRV, Degree)

    # Initialize and order domain based on some heuristics
    domain = initial_board.domains[(row, col)][:]
    domains_copy = {key: initial_board.domains[key][:] for key in initial_board.domains.keys()}
    domain = order_domain(initial_board, row, col, domain, domains_copy, LCV)

    # Go through each value in the domain, which is more or less a stack,
    # and assign it to the square then check if it's legal; 
    # if so, recurse; if not, unassign and return (to backtrack)

    # General backtracking algorithm: go through each value in the domain.
    # Assign it to the cell, then check if it's legal. If so, recurse and 
    # continue. If not, unassign and try the next one.

    while domain:
        #pdb.set_trace()
        d = domain.pop(0) 
        initial_board.set_value(row, col, d)

        global CHECKS
        CHECKS += 1

        if CHECKS % 1000 == 0:
            print "Tried %i variable assignments." % CHECKS

        #print "Looking for values for %i, %i; trying %i" % (row, col, d)

        # Forward checking algorithm: when we assign a value, check to see 
        # how it affects all of the domains of the other squares. If they
        # are empty, then we have an impossible move.

        if forward_checking:
            # Copy the board and the domains. We keep a separate, master 
            # copy of the domains that we /never/ modify, so that when we
            # mess with domains of board_copy we can reset them if needed.

            initial_board.change_domains(row, col)
            
            # Check the board to see if we found a square with an empty
            # domain. If so, reset the board and domain, and go to the 
            # next possible value in the domain ('continue').
            failure = False

            for i in range(size):
                for j in range(size):
                    if not initial_board.domains[(i, j)]:
                        failure = True
                        #print "Empty domain at %i, %i" % (i, j)
                        #pdb.set_trace()
                        initial_board.set_value(row, col, 0)
                        initial_board.domains = {key: domains_copy[key][:] for key in domains_copy.keys()}
                        #initial_board.domains = copy.deepcopy(domains_copy)
                        break

                if failure:
                    break

            if failure:
                continue

        # If we've reached a legal state, recurse and continue solving.
        if initial_board.is_legal():
            #print "Recursing"
            solve(initial_board, forward_checking, MRV, Degree, LCV)

        # If we reach here, we've returned -- either because it's complete
        # or because it's failed. If complete, continue returning. If failed,
        # set to 0 and keep going 
        global HAS_PRINTED
        
        if is_complete(initial_board):
            if not HAS_PRINTED:
                print CHECKS
                HAS_PRINTED = True

            return initial_board

        if CHECKS > 50000:
            if not HAS_PRINTED:
                print "Cut off at %i checks" % CHECKS
            HAS_PRINTED = True

            return initial_board

        initial_board.set_value(row, col, 0)
        initial_board.domains = {key: domains_copy[key][:] for key in domains_copy.keys()}

    # If we reached this point, we didn't find a legal value, so return
    return initial_board