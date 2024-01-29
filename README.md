## v1.1 Usage: 

### Create Transaction conducts a platform transaction
**POST stakeTransaction(codec, senderID, receiverAddress, amount)**
type - enum ['send', 'receive'] (buy, sell)
codec - string - specific to vendor
sender - [static] string - Sender ID (ask for this)
recipient - [static] string - Reciever Address
amount - number - e.g. 1.00
currency - enum - iso-4217 abbreviations

### New Muvor Wallet will retrieve all credentials for a new wallet (parameterless GET)
**GET createWallet()**