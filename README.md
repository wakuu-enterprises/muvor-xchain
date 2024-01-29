## v1.1 Usage: 

### Create Transaction conducts a platform transaction 
**POST stakeTransaction(codec, senderID, receiverAddress, amount)** 

* codec - ***string*** - _specific to vendor_
* sender - ***[static] string*** - _Sender ID (ask for this)_
* recipient - ***[static] string*** - _Reciever Address_
* amount - ***number*** - _e.g. 1.00_

### New Muvor Wallet will retrieve all credentials for a new wallet
**POST createWallet(user)**
* user - ***[static] string*** - _User ID (ask for this)_