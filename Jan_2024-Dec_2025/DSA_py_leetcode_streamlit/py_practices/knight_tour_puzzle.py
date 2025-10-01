import time
import os

class Chess_Exercise(object):
  def __init__(self):
    self.knight = "KN"
    self.enemy = "  "
    self.posible_movement = "[]"
    self.alredy_there = "--"

    self.posible_movements_knight = [
      [-1,-2],[ 1,-2],
      [ 2,-1],[ 2, 1],
      [ 1, 2],[-1, 2],
      [-2, 1],[-2,-1]
    ]
    self.posibilities_tree = []

  def create_starting_board(self):
    self.starts_position = [1,0]
    self.len_row = 3
    self.len_column = 8

    board = []
    
    for row in range(self.len_row):
      board.append([])
      for column in range(self.len_column):
          board[row].append(self.enemy)
    board[self.starts_position[0]][self.starts_position[1]] = self.knight

    return board
  
  def find_knight_position(self,board):
    position = []
    for row in board:
      if self.knight in row:
        position.append(board.index(row))
        position.append(row.index(self.knight))

        return position
        break
        
  def put_posible_movement_list(self,board,posible_movements):
    for movement in posible_movements:
      board[movement[0]][movement[1]] = self.posible_movement

  def knight_movement(self,board,new_position):
    old_position = self.find_knight_position(board)
    board[new_position[0]][new_position[1]] = self.knight
    board[old_position[0]][old_position[1]] = self.alredy_there

  def calculate_posible_movements(self,board):
    position = self.find_knight_position(board)
    posible_movements = []

    for movements_knight in self.posible_movements_knight:
      movement = [position[0] + movements_knight[0],
                  position[1] + movements_knight[1]]
      if self.len_row > movement[0] > -1:
        if self.len_column > movement[1] > -1:
          if not board[movement[0]][movement[1]] == self.alredy_there:
            posible_movements.append(movement)

    return posible_movements  

  def print_board(self,board):
    os.system("cls||clear")
    
    number_columns = 0
    number_rows = 0
    
    divider = "---+"
    string = "   |"
    
    for columns in range(self.len_column):
      divider += "----+"
      
      if number_columns == 10:
        number_columns = 0
      string += " " + str(number_columns) + "  |"
      
      number_columns += 1
    else:
      string += "\n"
      divider += "\n"
      string += divider

    
    for rows in board:
      if number_rows == 10:
        number_rows = 0
      row = " " + str(number_rows) + " |"
      for column in rows:
        row += " " + column + " |"
      string += row + "\n"
      string += divider

      number_rows += 1

    print(string)

  def copy_list(self,list):
    new_list = []
    for row in list:
      temporary_list = []
      for column in row:
        temporary_list.append(column)

      new_list.append(temporary_list)

    return new_list

  def calculate_total_movements(self,board):
    count = 0

    for row in board:
      for column in row:
        count += 1
          
    return count
    
  def making_posibilities_tree(self):
    starting_board = self.create_starting_board()
    self.posibilities_tree.append({"1":starting_board})

    total_movements = self.calculate_total_movements(starting_board)

    for section in self.posibilities_tree:
        
      for name,old_board in section.items():
        old_name = name
        posible_movements = self.calculate_posible_movements(old_board)
        if len(self.posibilities_tree) < total_movements:
          self.posibilities_tree.append({})
        if not len(posible_movements) == 0:
          pass
        posibility_counter = 1
        for movement in posible_movements:
          name = old_name + "." + str(posibility_counter)
          board = self.copy_list(old_board)
          self.knight_movement(board,movement)
        
          self.posibilities_tree[self.posibilities_tree.index(section) + 1][name] = board

          posibility_counter += 1

  def print_results(self):
    for name in self.posibilities_tree[len(self.posibilities_tree) - 1].keys():
      movement_count = 0
      string = ""
      for process in str(name):
        string += process
        if not process == ".":
          
          movement = self.posibilities_tree[movement_count][string]
          print(name,"\n")
          self.print_board(movement)
          input("")
          
          movement_count += 1

Knight_exercise = Chess_Exercise()
Knight_exercise.making_posibilities_tree()
Knight_exercise.print_results()