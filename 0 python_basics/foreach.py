fruits = ['apples','oranges','pears','bananas']
for fruit in fruits:
    print fruit + ' for sale'
    
fruitPrices = {'apples':2.00, 'oranges': 1.50, 'pears': 1.75}
for fruit, price in fruitPrices.items():
    if price < 2:
        print("{} cost {} a pound".format(fruit, price))
    else:
        print("{} are too expensive".format(fruit))