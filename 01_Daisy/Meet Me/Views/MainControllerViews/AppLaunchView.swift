//
//  AppLaunchView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 28.03.21.
//

import SwiftUI

struct AppLaunchView: View {
    var body: some View {
        VStack {
            
            Spacer()
            
            Text("DAISY")
                .font(.system(size: 75))
                .gradientForeground(gradient: secondaryGradient)
            Text("Nice to meet you.")
                .font(.title2)
                .padding(.bottom, 40)
            
            LoadingView(showLoadingScreen: .constant(true))
            
            Spacer()
            
            Text("Powered by StuPaNAK")
                .italic()
                .padding(.bottom, 45)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color("BackgroundMain"))
        .ignoresSafeArea()
    }
}

struct AppLaunchView_Previews: PreviewProvider {
    static var previews: some View {
        AppLaunchView()
    }
}
