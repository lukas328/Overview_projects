//
//  RingView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 11.03.21.
//

import SwiftUI

struct RingView: View {
    
    var progress: CGFloat
    var text: String
    
    var lineWidth: CGFloat = 5
    
    init(timeInMinutes: Int) {
        
        // handling the progress of the circle
        switch timeInMinutes {
        case 15:
            progress = 0.25
        case 30:
            progress = 0.5
        case 45:
            progress = 0.75
        case let x where x >= 60:
            progress = 1
        default:
            progress = 1
        }
        
        // handling the text output
        switch timeInMinutes {
        case let x where x < 60:
            text = String(timeInMinutes)
        case 60:
            text = "1h"
        case 90:
            text = "1,5"
        case 180:
            text = "2+"
        case let x where x > 60:
            text = "\(timeInMinutes / 60)h"
        default:
            text = String(timeInMinutes)
        }
        
    }
    
    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.black.opacity(0.1), style: StrokeStyle(lineWidth: lineWidth))
            
            
            Circle()
                .trim(from: 1 - progress, to: 1.0)
                .stroke(
                    AngularGradient(
                                gradient: Gradient(stops: [
                                    .init(color: Color(#colorLiteral(red: 0.0313725508749485, green: 0.45490196347236633, blue: 0.5490196347236633, alpha: 1)), location: 1),
                                    .init(color: Color(#colorLiteral(red: 0.0313725508749485, green: 0.45490196347236633, blue: 0.5490196347236633, alpha: 1)).opacity(0.2), location: 0)
                                ]
                                ),
                                center: UnitPoint(x: 0.5, y: 0.55)
                              ),
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .round, lineJoin: .round, miterLimit: .infinity, dash: [20, 0], dashPhase: 0)
                )
                .rotationEffect(Angle(degrees: 90))
                .rotation3DEffect(
                    Angle(degrees: 180),
                    axis: (x: 1, y: 0, z: 0)
                )
                .shadow(color: Color(#colorLiteral(red: 0.0313725508749485, green: 0.45490196347236633, blue: 0.5490196347236633, alpha: 1)).opacity(0.1), radius: 3, x: 0, y: 3)
            
            Text(text)
                .font(.system(size: 18))
                .bold()
        }
        .frame(width: 36, height: 36)
    }
}

struct RingView_Previews: PreviewProvider {
    static var previews: some View {
        RingView(timeInMinutes: 45)
    }
}
