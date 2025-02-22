import React from 'react';

const PaymentGatewayPage = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-4">
      <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-md">
        {/* Header */}
        <h2 className="text-xl font-semibold mb-4 text-black">Pay With UPI QR</h2>

        {/* QR Code Section */}
        <div className="flex items-center mb-6">
          <div className="border p-4 rounded-lg">
            <img 
              src="/sc.png" 
              alt="QR Code" 
              className="w-32 h-32" 
            />
          </div>
          <div className="ml-4">
            <p className="text-black">Scan the QR using any UPI app on your phone.</p>
            <div className="flex items-center space-x-2 mt-2">
              {/* Dummy icons for UPI apps */}
              {/* <img src="https://via.placeholder.com/20" alt="UPI app 1" className="w-6 h-6" />
              <img src="https://via.placeholder.com/20" alt="UPI app 2" className="w-6 h-6" />
              <img src="https://via.placeholder.com/20" alt="UPI app 3" className="w-6 h-6" />
              <img src="https://via.placeholder.com/20" alt="UPI app 4" className="w-6 h-6" /> */}
            </div>
            <p className="text-red-600 mt-2">QR Code is valid for few minutes only</p>
          </div>
        </div>

        {/* Preferred Payment Methods */}
        <h3 className="text-lg font-medium mb-2 text-black">Preferred Payment Methods</h3>
        <button className="w-full py-3 mb-4 bg-yellow-100 rounded-lg flex items-center justify-between px-4 hover:bg-yellow-200">
          <span className="flex items-center text-black">
            
            Pay using UPI
          </span>
          <span className="text-black">&rarr;</span>
        </button>

        {/* UPI, Cards & More */}
        <h3 className="text-lg font-medium mb-2 text-black">UPI, Cards & More</h3>
        <div className="space-y-3">
          <button className="w-full py-3 bg-gray-100 rounded-lg flex items-center justify-between px-4 hover:bg-gray-200">
            <span className="flex items-center text-black">
              
              UPI / QR
            </span>
            <span className="flex space-x-1">
              
            </span>
          </button>

          <button className="w-full py-3 bg-gray-100 rounded-lg flex items-center justify-between px-4 hover:bg-gray-200">
            <span className="flex items-center text-black">
              Card
            </span>
            <span className="flex space-x-1">
              
            </span>
          </button>

          <button className="w-full py-3 bg-gray-100 rounded-lg flex items-center justify-between px-4 hover:bg-gray-200">
            <span className="flex items-center text-black">
              
              Netbanking
            </span>
            <span className="flex space-x-1">
           
            </span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default PaymentGatewayPage;
