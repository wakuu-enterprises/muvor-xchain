## v1.1 Usage: 

### Create Transaction conducts a platform transaction 
**POST stakeTransaction(codec, senderID, receiverAddress, amount)** 

- > type - ***enum*** _['send', 'receive'] (buy, sell)_
- > codec - ***string*** - _specific to vendor_
- > sender - ***[static] string*** - _Sender ID (ask for this)_
- > recipient - ***[static] string*** - _Reciever Address_
- > amount - ***number*** - _e.g. 1.00_
- > currency - ***enum*** - _iso-4217 abbreviations_

### New Muvor Wallet will retrieve all credentials for a new wallet (parameterless GET)
**GET createWallet()**