import time, os

execfile('SudokuStarter.py')

d = 'input_puzzles/more/small/'
for i, f in enumerate(os.listdir(d)):
	start = time.clock()
	sb = init_board(d + f)
	sb.print_board()
	fb = solve(sb, forward_checking = False, MRV = False, Degree = False, LCV = False)
	print 'Board #%i was %s' % (i + 1, f)
	print time.clock() - start
	fb.print_board()

'''
start = time.clock()
#sb = init_board('input_puzzles/more/25x25/25x25.1.sudoku')
#sb = init_board('input_puzzles/more/16x16/16x16.2.sudoku')
sb = init_board('input_puzzles/easy/9_9.sudoku')
sb.print_board()
fb = solve(sb, forward_checking = True, MRV = False, Degree = True, LCV = False)
print time.clock() - start
fb.print_board()
'''