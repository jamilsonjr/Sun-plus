pragma solidity >=0.7.0 <0.9.0;

contract SunPlus
{
    enum StateType { 
      ChargerAvailable,
      ChargerReserved,
      ChargingEvent,
      Payment
    }

    address public InstanceOwner;
    string public Description;
    int public AskingPrice;
    StateType public State;

    address public InstanceBuyer;


    constructor(string memory description, int price) 
    {
        InstanceOwner = msg.sender;
        AskingPrice = price;
        Description = description;
        State = StateType.ChargerAvailable;
    }

    function PlaceOrder() public
    {
        if (State != StateType.ChargerAvailable)
        {
            revert();
        }
        
        if (InstanceOwner == msg.sender)
        {
            revert();
        }

        InstanceBuyer = msg.sender;
        State = StateType.ChargerReserved;
    }

    function Reject() public
    {
        if ( State != StateType.ChargerReserved )
        {
            revert();
        }

        if (InstanceOwner != msg.sender)
        {
            revert();
        }

        InstanceBuyer = 0x0000000000000000000000000000000000000000;
        State = StateType.ChargerAvailable;
    }

    function Start() public
    {
        if (InstanceOwner == msg.sender)
        {
            revert();
        }
        State = StateType.ChargingEvent;
    }
    function FinishCharging() public
    {
        if (State != StateType.ChargingEvent)
        {
            revert();
        }

        if (InstanceOwner == msg.sender)
        {
            revert();
        }
        
        InstanceBuyer = msg.sender;
        State = StateType.Payment;
    }
    
    function MakeChargerAvailable() public
    {
        if (State != StateType.Payment)
        {
            revert();
        }
         if ( msg.sender != InstanceOwner )
        {
            revert();
        }
        
        State = StateType.ChargerAvailable;
    }
}